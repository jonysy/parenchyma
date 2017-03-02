use std::{cmp, convert, mem, os, ptr};
use std::collections::HashMap;
use super::{cl, hl};
use super::super::super::{Buffer, Device, Error, Result};

/// Provides the OpenCL framework.
#[derive(Debug)]
pub struct OpenCl {
    /// List of available devices for the first platform found.
    pub available_devices: Vec<OpenClDevice>,
}

impl OpenCl {
    /// The name of the framework.
    const FRAMEWORK_NAME: &'static str = "OPEN_CL";

    /// Initializes the framework.
    pub fn try_new() -> hl::Result<Self> {

        let available_devices = {
            hl::platforms()?.remove(0).devices()?.into_iter()
                .map(|raw| unsafe {
                    OpenClDevice {
                        raw, 
                        context: mem::uninitialized(), 
                        queue: mem::uninitialized(),
                    }
                }).collect()
        };

        Ok(OpenCl { available_devices })
    }
}

/// A context is responsible for managing OpenCL objects and resources.
///
/// Contexts are used by the OpenCL runtime for managing objects such as command-queues,
/// memory, program and kernel objects and for executing kernels on one or more devices
/// specified in the context. A single context for multiple devices, a single context for a 
/// single device, and a context for each device are all possible.
///
/// Memory objects can be copied to host memory, from host memory, or to other memory objects.
/// Copying from the host to a device is considered _writing_. Copying from a device to the host is
/// considered _reading_.
///
/// # Notes
///
/// * The context is _shared_ if `selection` > 1.
/// * Multi-platform contexts are not supported in OpenCL.
//
// TODO prevent mixing devices from different contexts
#[derive(Clone, Debug)]
pub struct OpenClContext {
    raw: hl::Context,
    selection: Vec<OpenClDevice>,
    program: hl::Program,
    kernels: HashMap<&'static str, hl::Kernel>,
}

impl OpenClContext {

    /// Constructs a context from a `selection` of devices.
    pub fn new(mut selection: Vec<OpenClDevice>) -> hl::Result<OpenClContext> {
        let raw_devices: Vec<hl::Device> = selection.iter().map(|d| d.raw.clone()).collect();
        let raw = hl::Context::new(&raw_devices)?;

        for device in selection.iter_mut() {
            let queue = hl::Queue::new(&raw, &device.raw, 0)?;

            device.context = raw.clone();
            device.queue = queue;
        }

        let source = vec![include_str!("source/math.cl")];
        let program = raw.create_program_with_source(&source)?;
        program.build(&raw_devices, None)?;

        let mut kernels = HashMap::new();
        kernels.insert("sigmoid_f32", program.create_kernel("array_sigmoid_f32")?);

        Ok(OpenClContext { raw, selection, program, kernels })
    }
}

#[derive(Clone, Debug)]
pub struct OpenClDevice {
    raw: hl::Device,
    context: hl::Context,
    /// A command queue
    ///
    /// A command queue is the mechanism for interaction with the device. The queue is used for 
    /// operations such as kernel launches and memory copies. At least one command queue per device
    /// is required. Queues are used by the host application to submit work to devices and 
    /// associated with devices within a context.
    ///
    /// Commands:
    ///
    /// - Memory copy or mapping
    /// - Device code execution
    /// - Synchronization point
    ///
    /// Modes:
    ///
    /// - In-order
    /// - Out-of-order
    ///
    /// ## Note
    ///
    /// * Use events to synchronize
    queue: hl::Queue,
}

impl<T> Device<T> for OpenClDevice {

    fn allocate(&self, size: usize, host: Option<&mut [T]>) -> Result<Buffer> {
        let buffer  = match host {
            Some(host_data) => {
                let flag = cl::CL_MEM_COPY_HOST_PTR;
                let pointer = host_data.as_mut_ptr() as *mut os::raw::c_void;

                self.context.create_buffer(flag, size, pointer)?
            },

            _ => {
                //  TODO
                let flag = cl::CL_MEM_READ_WRITE;
                self.context.create_buffer(flag, size, None)?
            }
        };

        Ok(Buffer::OpenCl(buffer))
    }

    fn synch_in(&self, buffer: &mut Buffer, source: &[T]) -> Result {

        unimplemented!()
    }

    fn synch_out(&self, buffer: &Buffer, destination: &mut [T]) -> Result {

        unimplemented!()
    }
}

impl cmp::Eq for OpenClDevice { }

impl cmp::PartialEq for OpenClDevice {

    fn eq(&self, other: &Self) -> bool {
        self.context == other.context && self.raw == other.raw
    }
}

impl convert::From<hl::Error> for Error {

    fn from(hl_error: hl::Error) -> Error {

        unimplemented!()
    }
}
//! OpenCL backend support for Parenchyma.
//!
//! ## Terminology
//!
//! Work-group: A collection of work items + has a unique work-group ID. work-groups are assigned
//! to execute on compute-units
//! Work-item: An instance of a kernel at run time + has a unique ID within the work-group
//!
//! ## Flow
//!
//! - Initialize the framework
//! - Select the desired platform
//! - Select the desired devices from the platform
//! - Create a context 
//! - create a command queue per device
//! - Compile programs
//!     - A program is essentially a collection of kernels.
//! - Create a kernel from the successfully compiled program
//!     - A kernel is the smallest unit of execution. Kernels are expensive to start so they're
//!     typically used to do a large amount of work.
//! - Specify arguments to the kernel
//! - Allocate memory on devices
//! - Transfer data to devices
//! - Execute
//! - Transfer results back
//! - Free memory on devices
//!
//! ## Events (TODO)
//!
//! Most OpenCL operations happen asynchronously on the OpenCL Device. To provide the possibility 
//! to order and synchronize multiple operations, the execution of an operation yields a event 
//! object. This event can be used as an input to other operations which will wait until this event 
//! has finished executing to run.

pub mod cl;
pub mod hl;

use std::{cmp, convert, os, ptr};
use std::collections::HashMap;
use super::super::{Buffer, Device, Error, ErrorKind, Result};

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
        // TODO return Err if `platforms` || `devices` == 0
        let available_devices = {
            hl::platforms()?.remove(0).devices()?.into_iter()
                .map(|raw| OpenClDevice {
                        raw, 
                        context: None, 
                        queue: None,
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
    //
    // TODO replace with `TryFrom::try_from` when stable..
    pub fn try_from(mut selection: Vec<OpenClDevice>) -> hl::Result<OpenClContext> {
        let raw_devices: Vec<hl::Device> = selection.iter().map(|d| d.raw.clone()).collect();
        let raw = hl::Context::new(&raw_devices)?;

        for device in selection.iter_mut() {

            let queue = hl::Queue::new(&raw, &device.raw, 0)?;

            device.context = Some(raw.clone());
            device.queue = Some(queue);
        }

        let source = vec![include_str!("source/math.cl")];
        let program = raw.create_program_with_source(&source)?;
        program.build(&raw_devices, None)?;

        let mut kernels = HashMap::new();
        let kernel = program.create_kernel("array_sigmoid_f32")?;
        kernels.insert("sigmoid_f32", kernel);

        Ok(OpenClContext { raw, selection, program, kernels })
    }

    pub fn selection(&self) -> &[OpenClDevice] {
        &self.selection
    }

    pub fn kernels(&self) -> &HashMap<&'static str, hl::Kernel> {
        &self.kernels
    }
}

#[derive(Clone, Debug)]
pub struct OpenClDevice {
    raw: hl::Device,
    context: Option<hl::Context>,
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
    queue: Option<hl::Queue>,
}

impl OpenClDevice {

    pub fn queue(&self) -> &hl::Queue {
        self.queue.as_ref().unwrap()
    }

    pub fn context(&self) -> &hl::Context {
        self.context.as_ref().unwrap()
    }
}

impl<T> Device<T> for OpenClDevice where T: Clone {

    fn allocate(&self, size: usize, host: Option<&mut [T]>) -> Result<Buffer> {
        let buffer  = match host {
            Some(host_data) => {
                let flag = cl::CL_MEM_READ_ONLY | cl::CL_MEM_COPY_HOST_PTR;
                let pointer = host_data.as_mut_ptr() as *mut os::raw::c_void;

                // create buffer and fill it immediately
                self.context().create_buffer(flag, size, pointer)?
            },

            _ => {
                //  TODO
                let flag = cl::CL_MEM_READ_WRITE;
                self.context().create_buffer(flag, size, None)?
            }
        };

        Ok(Buffer::OpenCl(buffer))
    }

    fn synch_in(&self, buffer: &mut Buffer, source: &[T]) -> Result {
        // let bu = buffer.as_opencl();
        // let len = source.len();
        // let ptr = source.as_ptr();
        // self.queue().enqueue_write_buffer(bu, true, 0, len, ptr as _, &[])?;
        // Ok(())
        unimplemented!()
    }

    fn synch_out(&self, buffer: &Buffer, destination: &mut [T], size: usize) -> Result {
        let bu = buffer.as_opencl();
        let len = destination.len();
        let ptr = destination.as_ptr();
        self.queue().enqueue_read_buffer(bu, true, 0, ::std::mem::size_of::<T>() * len, ptr as _, &[])?;
        Ok(())
    }
}

impl cmp::Eq for OpenClDevice { }

impl cmp::PartialEq for OpenClDevice {

    fn eq(&self, other: &Self) -> bool {
        self.context == other.context && self.raw == other.raw
    }
}

impl convert::From<hl::Error> for Error {

    fn from(e: hl::Error) -> Error {

        Error::new(ErrorKind::Framework { name: OpenCl::FRAMEWORK_NAME }, e)
    }
}
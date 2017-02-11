use opencl::api::ContextPtr;
use parenchyma::{Context, NativeContext, NativeMemory};
use std::hash::{Hash, Hasher};
use super::{OpenCLDevice, OpenCLError, OpenCLMemory, OpenCL, OpenCLQueue};

// notes:
// shared context if more than one device is passed in

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
#[derive(Clone, Debug)]
pub struct OpenCLContext {
    ptr: ContextPtr,
    /// <sup>*</sup>Multi-platforms contexts are not supported in OpenCL.
    // platform_id: core::PlatformId,
    selected_devices: Vec<OpenCLDevice>,
    // /// A queue is used by the host application to submit work to a device.
    // queues: Vec<OpenCLQueue>,
}

impl Context for OpenCLContext {
    /// The framework associated with this context.
    type F = OpenCL;

    /// Constructs a context from a selection of devices.
    fn new(devices: Vec<OpenCLDevice>) -> Result<Self, OpenCLError> {

        let ndevices = devices.len();

        match ndevices {
            0 => unimplemented!(),
            1 | 2 => {
                let device_ptrs = &vec![devices[0].ptr.clone()];
                let ptr = ContextPtr::new(&device_ptrs)?;

                Ok(OpenCLContext {
                    ptr: ptr,
                    selected_devices: devices,
                })
            },
            _ => unimplemented!(),
        }
    }

    /// Allocates memory
    fn allocate_memory(&self, _: usize) -> Result<OpenCLMemory, OpenCLError> {

        unimplemented!()
    }

    fn synch_in(&self, _: &mut OpenCLMemory, _: &NativeContext, _: &NativeMemory) -> Result<(), OpenCLError> {

        unimplemented!()
    }

    fn synch_out(&self, _: &OpenCLMemory, _: &NativeContext, _: &mut NativeMemory) -> Result<(), OpenCLError> {

        unimplemented!()
    }
}

impl PartialEq for OpenCLContext {

    fn eq(&self, other: &Self) -> bool {
        //self.id == other.id
        unimplemented!()
    }
}

impl Eq for OpenCLContext { }

impl Hash for OpenCLContext {

    fn hash<H>(&self, state: &mut H) where H: Hasher {
        // (unsafe { self.id.as_ptr() as isize }).hash(state);
        unimplemented!()
    }
}
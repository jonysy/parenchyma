use opencl::api::{ContextPtr, QueuePtr};
use parenchyma::{Context, NativeContext, NativeMemory};
use std::hash::{Hash, Hasher};
use super::{OpenCLDevice, OpenCLError, OpenCLMemory, OpenCL, OpenCLQueue};

// notes:
// shared context if more than one device is passed in
// Multi-platforms contexts are not supported in OpenCL.

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
    selected_device: OpenCLDevice,
    queue: OpenCLQueue,
}

impl Context for OpenCLContext {
    /// The framework associated with this context.
    type F = OpenCL;

    /// Constructs a context from a selection of devices.
    fn new(device: OpenCLDevice) -> Result<Self, OpenCLError> {

        let device_ptr_vec = &vec![device.ptr.clone()];
        let ptr = ContextPtr::new(&device_ptr_vec)?;
        let queue = OpenCLQueue { ptr: QueuePtr::new(&ptr, &device.ptr, 0)? };

        Ok(OpenCLContext { ptr: ptr, selected_device: device, queue: queue })
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
use opencl::{ContextPtr, MemoryObject, QueuePtr};
use opencl::enqueue;
use parenchyma::{Context, NativeContext, NativeMemory};
use std::hash::{Hash, Hasher};

use super::{OpenCLDevice, OpenCLError, OpenCLMemory, OpenCL, OpenCLQueue, Result};

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
    fn new(device: OpenCLDevice) -> Result<Self> {

        let device_ptr_vec = &vec![device.ptr.clone()];
        let ptr = ContextPtr::new(&device_ptr_vec)?;
        let queue = OpenCLQueue { ptr: QueuePtr::new(&ptr, &device.ptr, 0)? };

        Ok(OpenCLContext { ptr: ptr, selected_device: device, queue: queue })
    }

    /// Allocates memory
    fn allocate_memory(&self, size: usize) -> Result<OpenCLMemory> {

        let mem_obj = MemoryObject::create_buffer(&self.ptr, size)?;

        Ok(OpenCLMemory { obj: mem_obj })
    }

    fn synch_in(&self, destn: &mut OpenCLMemory, _: &NativeContext, src: &NativeMemory) -> Result {

        let s_size = src.size();
        let s_ptr = src.as_slice().as_ptr();

        let _ = enqueue::write_buffer(&self.queue.ptr, &destn.obj, true, 0, s_size, s_ptr, &[])?;

        Ok(())
    }

    fn synch_out(&self, src: &OpenCLMemory, _: &NativeContext, destn: &mut NativeMemory) -> Result {

        let d_size = destn.size();
        let d_ptr = destn.as_mut_slice().as_mut_ptr();

        let _ = enqueue::read_buffer(&self.queue.ptr, &src.obj, true, 0, d_size, d_ptr, &[])?;

        Ok(())
    }
}

impl PartialEq for OpenCLContext {

    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl Eq for OpenCLContext { }

// impl Hash for OpenCLContext {

//     fn hash<H>(&self, state: &mut H) where H: Hasher {
//         // (unsafe { self.id.as_ptr() as isize }).hash(state);
//         unimplemented!()
//     }
// }
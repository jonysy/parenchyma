use {Alloc, ComputeDevice, ErrorKind, Memory, Result, Shape, Synch, Viewable};
use std::os::raw::c_void;
use super::OpenCLMemory;
use super::super::{foreign, high};
use utility;

/// Represents an OpenCL device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenCLDevice {
    pub(super) device: high::Device,
    pub(super) context: high::Context,
    /// A command queue
    ///
    /// A command queue is the mechanism for interaction with the device. The queue is used for 
    /// operations such as kernel launches and memory copies. At least one command queue per device
    /// is required. Queues are used by the host application to submit work to devices and 
    /// associated with devices within a context.
    ///
    /// __commands__:
    ///
    /// - memory copy or mapping
    /// - device code execution
    /// - synchronization point
    ///
    /// __modes__:
    ///
    /// - in-order
    /// - out-of-order
    ///
    /// ## TODO
    ///
    /// * Use events to synchronize
    pub(super) queue: high::Queue,
}

impl OpenCLDevice {

    /// Returns the OpenCL command queue.
    ///
    /// [package author]
    pub fn queue(&self) -> &high::Queue {
        &self.queue
    }
}

impl Viewable for OpenCLDevice {

    fn view(&self) -> ComputeDevice {
        ComputeDevice::OpenCL(self.clone())
    }
}

impl<T> Alloc<T> for OpenCLDevice {


    fn alloc(&self, shape: &Shape) -> Result<Memory<T>> {
        // TODO

        let flag = foreign::CL_MEM_READ_WRITE;
        let length = shape.capacity();
        let size = utility::allocated::<T>(length);
        let buffer = self.context.create_buffer(flag, size, None)?;
        let cl_buffer = OpenCLMemory { buf: buffer };
        Ok(Memory::OpenCL(cl_buffer))
    }

    fn allocwrite(&self, shape: &Shape, mut data: Vec<T>) -> Result<Memory<T>> {
        // TODO

        let flag = foreign::CL_MEM_READ_ONLY | foreign::CL_MEM_COPY_HOST_PTR;
        let length = shape.capacity();
        let size = utility::allocated::<T>(length);
        let pointer = data.as_mut_ptr();

        // create buffer and fill it immediately
        let buffer = self.context.create_buffer(flag, size, pointer as *mut c_void)?;
        let cl_buffer = OpenCLMemory { buf: buffer };
        Ok(Memory::OpenCL(cl_buffer))
    }
}

impl<T> Synch<T> for OpenCLDevice {

    fn write(&self, memory: &mut Memory<T>, _: &ComputeDevice, source: &Memory<T>) -> Result {
        match *source {
            Memory::Native(ref native_memory) => {
                let cl_memory = unsafe { memory.as_opencl_unchecked() };

                let length = native_memory.len();
                let size = utility::allocated::<T>(length);
                let slice = native_memory.as_slice_memory_order().unwrap();
                let slice_pointer = slice.as_ptr();

                let ref buf = cl_memory.buf;
                let block = true; // TODO async
                let offset = 0;
                let _ = self.queue
                    .enqueue_write_buffer(buf, block, offset, size, slice_pointer as *const c_void, &[])?;

                Ok(())
            },

            _ => Err(ErrorKind::NoAvailableSynchronizationRouteFound.into()),
        }
    }

    /// Synchronizes `memory` to `destination`.
    fn read(&self, memory: &Memory<T>, _: &mut ComputeDevice, destination: &mut Memory<T>) -> Result {
        match *destination {
            Memory::Native(ref mut native_memory) => {
                let cl_memory = unsafe { memory.as_opencl_unchecked() };

                let length = native_memory.len();
                let size = utility::allocated::<T>(length);
                let slice = native_memory.as_slice_memory_order_mut().unwrap();
                let slice_pointer = slice.as_mut_ptr();

                let ref buf = cl_memory.buf;
                let block = true; // TODO async
                let offset = 0;
                let _ = self.queue
                    .enqueue_read_buffer(buf, block, offset, size, slice_pointer as *mut c_void, &[])?;

                Ok(())
            },

            _ => Err(ErrorKind::NoAvailableSynchronizationRouteFound.into()),
        }
    }
}
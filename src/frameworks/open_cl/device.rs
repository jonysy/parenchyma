use ocl;

use super::{OpenCLBuf, OpenCLMemory};
use super::super::super::compute_device::{Allocate, ComputeDevice};
use super::super::super::error::Result;
use super::super::super::memory::Memory;
use super::super::super::tensor::{TensorShape, TensorType};

/// Represents an Open CL device.
#[derive(Clone, Debug)]
pub struct OpenCLDevice {
    pub(in frameworks::open_cl) device: ocl::Device,
    pub(in frameworks::open_cl) context: ocl::Context,
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
    pub(in frameworks::open_cl) queue: ocl::Queue,
}

impl OpenCLDevice {
    pub fn queue(&self) -> &ocl::Queue {
        &self.queue
    }
}

impl ComputeDevice for OpenCLDevice { }

impl<T> Allocate<T> for OpenCLDevice  where T: TensorType + 'static {
    fn allocate(&self, shape: &TensorShape) -> Result<Box<Memory<T>>> {
        let ctx = &self.context;
        let flags_opt = Some(ocl::flags::MEM_READ_WRITE);
        let dims = ocl::SpatialDims::One(shape.capacity);
        let host_data = None;
        let buf: OpenCLBuf<T> = OpenCLBuf { 
            buf: ocl::Buffer::new(ctx, flags_opt, dims, host_data)? 
        };
        let device = self.clone();
        let memory = Box::new(OpenCLMemory {
            buf,
            device,
        });

        return Ok(memory);
    }
}
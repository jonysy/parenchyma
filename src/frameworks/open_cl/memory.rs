use ocl;
use super::OpenCLDevice;
use super::super::NativeMemory;
use super::super::super::compute_device::ComputeDevice;
use super::super::super::error::{ErrorKind, Result};
use super::super::super::memory::{Memory, TransferDirection};
use super::super::super::tensor::TensorType;

/// A `Memory` wraps around an OpenCL buffer id that manages its deallocation, named 
/// as such for consistency's sake.
///
/// Memory objects can be copied to host memory, from host memory, or to other memory objects.
/// Copying from the host to a device is considered _writing_. Copying from a device to the host is
/// considered _reading_.
///
/// Unlike CUDA, OpenCL [buffers][1] are only context specific, not device specific. Also note: 
/// currently, lazy allocation is used on the NVIDIA driver. That is, the buffer object, in a sense,
/// is located _nowhere_ when allocated. It only exists when needed.
///
/// [1]: https://goo.gl/S9B3TL
#[derive(Clone, Debug)]
pub struct OpenCLBuf<T> where T: TensorType {
    pub(in super) buf: ocl::Buffer<T>,
}

/// Memory representation for Open CL 
pub struct OpenCLMemory<T> where T: TensorType {
    pub(in super) buf: OpenCLBuf<T>,
    pub(in super) device: OpenCLDevice,
}

impl<T> Memory<T> for OpenCLMemory<T> where T: TensorType + 'static {
    fn synchronized(&self, device: &ComputeDevice) -> bool {
        if let Some(op) = device.downcast_ref::<OpenCLDevice>() {
            (self.device.device == op.device) && (self.device.context.core() == op.context.core())
        } else {
            false
        }
    }

    fn transfer(&mut self, dir: TransferDirection, m: &mut Memory<T>) -> Result {
        match dir {
            TransferDirection::TransferIn => {
                if let Some(na) = m.downcast_ref::<NativeMemory<T>>() {
                    let buffer_write_cmd = unsafe {
                        self.buf.buf.write(
                            na.0.as_slice_memory_order()
                                .expect("the array's data is not contiguous") // TODO
                        )
                        .queue(&self.device.queue)
                        .block(true) // TODO
                        .len(na.0.len())
                    };

                    Ok(buffer_write_cmd.enq()?)
                } else {
                    Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
                }
            },

            TransferDirection::TransferOut => {
                if let Some(na) = m.downcast_mut::<NativeMemory<T>>() {
                    let length = na.0.len();

                    let buffer_read_cmd = unsafe {
                        self.buf.buf.read(
                            na.0.as_slice_memory_order_mut()
                                .expect("the array's data is not contiguous") // TODO
                        )
                        .queue(&self.device.queue)
                        .block(true) // TODO
                        .len(length)
                    };

                    Ok(buffer_read_cmd.enq()?)
                } else {
                    Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
                }
            }
        }
    }
}

impl<T: TensorType> ::ocl::core::AsMem<T> for OpenCLMemory<T> {
    fn as_mem(&self) -> &::ocl::core::Mem {
        self.buf.buf.as_mem()
    }
}

unsafe impl<T: TensorType> ::ocl::core::MemCmdAll for OpenCLMemory<T> { }
unsafe impl<'a, T: TensorType> ::ocl::core::MemCmdAll for &'a OpenCLMemory<T> { }
unsafe impl<'a, T: TensorType> ::ocl::core::MemCmdAll for &'a mut OpenCLMemory<T> { }
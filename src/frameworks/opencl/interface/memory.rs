use Memory;
use super::super::high;

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
pub struct OpenCLMemory {
    pub(super) buf: high::Buffer,
}

impl<T> ::opencl::high::KernelArg for Memory<T> {
    fn size(&self) -> usize { ::std::mem::size_of::<::opencl::foreign::cl_mem>() }

    fn pointer(&self) -> ::opencl::foreign::cl_mem {

        unsafe { self.as_opencl_unchecked().buf.pointer() }
    }
}
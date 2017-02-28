use super::native::NativeContext;
use super::opencl::OpenCLContext;

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
#[derive(Clone, Debug)]
pub enum ContextView {
    Native(NativeContext),
    OpenCL(OpenCLContext),
}

impl ContextView {

    pub fn as_native(&self) -> Option<&NativeContext> {
        match *self {
            ContextView::Native(ref native) => Some(native),
            _ => None,
        }
    }
    
    pub fn as_mut_native(&mut self) -> Option<&mut NativeContext> {
        match *self {
            ContextView::Native(ref mut native) => Some(native),
            _ => None,
        }
    }

    pub fn as_opencl(&self) -> Option<&OpenCLContext> {
        match *self {
            ContextView::OpenCL(ref opencl) => Some(opencl),
            _ => None,
        }
    }

    pub fn as_mut_opencl(&mut self) -> Option<&mut OpenCLContext> {
        match *self {
            ContextView::OpenCL(ref mut opencl) => Some(opencl),
            _ => None,
        }
    }
}
use super::native::NativeContext;
use super::opencl::OpenCLContext;

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
#[derive(Clone, Debug)]
pub enum ContextView {
    Native(NativeContext),
    OpenCL(OpenCLContext),
}
use super::opencl;

#[derive(Debug)]
pub enum Buffer {
    /// A "newtype" around an OpenCL memory id that manages its deallocation.
    OpenCl(opencl::hl::Buffer),
}
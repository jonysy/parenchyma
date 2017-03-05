use super::opencl;

#[derive(Clone, Debug)]
pub enum Buffer {
    /// A "newtype" around an OpenCL memory id that manages its deallocation.
    OpenCl(opencl::hl::Buffer),
}

impl Buffer {

    pub fn as_opencl(&self) -> &opencl::hl::Buffer {
        match *self {
            Buffer::OpenCl(ref opencl) => opencl,
        }
    }
}
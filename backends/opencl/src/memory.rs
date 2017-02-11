use opencl::api::MemoryObject;

/// Holds a OpenCL memory id and manages its deallocation
#[derive(Debug)]
pub struct OpenCLMemory {
    pub(super) obj: MemoryObject,
}
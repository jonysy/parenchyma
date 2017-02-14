use api;

/// CUDA memory
pub struct CudaMemory {
    pub(super) dptr: api::Memory,
}
use cuda;

/// CUDA memory
pub struct CudaMemory {
    pub(super) dptr: cuda::Memory,
}
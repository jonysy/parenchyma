//! CUDA backend support for Parenchyma.
//!
//! * [NVIDIA CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! * [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

pub use self::context::CudaContext;
pub use self::device::CudaDevice;
pub use self::framework::Cuda;
pub use self::memory::CudaMemory;
pub use self::result::Result;

mod api;
mod context;
mod device;
mod framework;
mod memory;
mod result;
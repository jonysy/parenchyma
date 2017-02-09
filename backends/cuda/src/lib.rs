//! CUDA backend support for Parenchyma.
//!
//! * [NVIDIA CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! * [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

#![allow(warnings)]
#![feature(associated_consts, pub_restricted, type_ascription)]

extern crate cuda;
extern crate parenchyma;

pub use self::context::CudaContext;
pub use self::device::CudaDevice;
pub use self::error::CudaError;
pub use self::framework::Cuda;
pub use self::memory::CudaMemory;

mod context;
mod device;
mod error;
mod framework;
mod memory;
//! * [NVIDIA CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! * [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

#![feature(untagged_unions)]

extern crate parenchyma;

pub use self::device::Device;
pub use self::framework::Cuda;

mod api;
mod device;
mod error;
mod framework;
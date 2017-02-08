//! * [NVIDIA CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! * [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

#![allow(warnings)]
#![feature(untagged_unions)]

extern crate cuda;
extern crate parenchyma;

pub use self::device::Device;
pub use self::framework::Cuda;

mod device;
mod framework;
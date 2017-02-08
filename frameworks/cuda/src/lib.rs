//! * [NVIDIA CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! * [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

#![allow(warnings)]
#![feature(associated_consts, pub_restricted)]

extern crate cuda;
extern crate parenchyma;

pub use self::context::Context;
pub use self::device::Device;
pub use self::framework::Cuda;
pub use self::memory::Memory;

mod context;
mod device;
mod framework;
mod memory;
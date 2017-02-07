//! * [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
//! * [NVIDIA CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)

#![feature(untagged_unions)]

extern crate parenchyma;

mod core;
mod error;
#[allow(warnings)]
mod sys;
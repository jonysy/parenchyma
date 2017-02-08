//! * [The OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)

#![feature(associated_consts, pub_restricted)]

extern crate cl_sys as sys;
extern crate ocl as cl;
extern crate ocl_core as core;
extern crate parenchyma;

pub use self::context::OpenCLContext;
pub use self::device::OpenCLDevice;
pub use self::error::OpenCLError;
pub use self::framework::OpenCL;
pub use self::memory::OpenCLMemory;
pub use self::platform::OpenCLPlatform;
pub use self::queue::OpenCLQueue;

mod context;
mod device;
mod error;
mod framework;
mod memory;
mod platform;
mod queue;
//! * [The OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)

#![feature(associated_consts, pub_restricted)]

extern crate cl_sys as sys;
extern crate ocl as cl;
extern crate ocl_core as core;
extern crate parenchyma;

pub use self::context::Context;
pub use self::device::Device;
pub use self::framework::OpenCL;
pub use self::memory::Memory;
pub use self::platform::Platform;
pub use self::queue::Queue;

mod context;
mod device;
mod framework;
mod memory;
mod platform;
mod queue;
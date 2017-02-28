//! * [Specification](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)
//! * [Reference Card](https://www.khronos.org/files/opencl-1-1-quick-reference-card.pdf)

#[macro_use]
mod macros;

pub mod error;
pub mod sys;

pub use self::context::Context;
pub use self::device::Device;
pub use self::event::Event;
pub use self::kernel::Kernel;
pub use self::memory::Memory;
pub use self::platform::Platform;
pub use self::program::Program;
pub use self::queue::Queue;

mod context;
mod device;
mod event;
mod kernel;
mod memory;
mod platform;
mod program;
mod queue;

/// Number of platforms
pub fn nplatforms() -> error::Result<u32> {

    unsafe {

        let mut nplatforms = 0;

        result!(sys::clGetPlatformIDs(0, ::std::ptr::null_mut(), &mut nplatforms)
            => Ok(nplatforms))
    }
}

/// Obtain the list of platforms available.
pub fn platform_ids() -> error::Result<Vec<Platform>> {

    let nplatforms = nplatforms()?;

    unsafe {
        
        let mut vec_id = vec![0 as sys::cl_platform_id; nplatforms as usize];

        result!(sys::clGetPlatformIDs(nplatforms, vec_id.as_mut_ptr(), ::std::ptr::null_mut())
            => Ok(vec_id.iter().map(|&id| Platform(id)).collect()))
    }
}
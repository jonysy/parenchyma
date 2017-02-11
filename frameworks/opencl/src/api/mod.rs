pub use self::context::ContextPtr;
pub use self::device::DevicePtr;
pub use self::memory::MemoryPtr;
pub use self::platform::PlatformPtr;
pub use self::queue::QueuePtr;

mod context;
mod device;
mod memory;
mod platform;
mod queue;

use opencl_sys;
use std::ptr;

use error::{ErrorKind, Result};

/// Number of platforms
pub fn nplatforms() -> Result<u32> {
    unsafe {
        let mut nplatforms = 0;

        match opencl_sys::clGetPlatformIDs(0, ptr::null_mut(), &mut nplatforms) {
            opencl_sys::CLStatus::CL_SUCCESS => Ok(nplatforms),

            e @ _ => Err((e.into(): ErrorKind).into())
        }
    }
}

/// Obtain the list of platforms available.
pub fn platform_ids() -> Result<Vec<PlatformPtr>> {

    let nplatforms = nplatforms()?;

    unsafe {
        let mut vec_id = vec![0 as opencl_sys::cl_platform_id; nplatforms as usize];

        match opencl_sys::clGetPlatformIDs(nplatforms, vec_id.as_mut_ptr(), ptr::null_mut()) {
            opencl_sys::CLStatus::CL_SUCCESS => {
                Ok(vec_id.iter().map(|&id| PlatformPtr(id)).collect())
            },

            e @ _ => Err((e.into(): ErrorKind).into())
        }
    }
}
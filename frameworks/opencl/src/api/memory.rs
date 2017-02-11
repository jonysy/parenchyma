use opencl_sys;
use std::ptr;

use error::{ErrorKind, Result};
use super::ContextPtr;

#[derive(Debug)]
pub struct MemoryObject(pub(super) opencl_sys::cl_mem);

impl MemoryObject {

    /// Creates a buffer object.
    pub fn create_buffer(context: &ContextPtr, size: usize) -> Result<MemoryObject> {

        unsafe {
            
            let flags = opencl_sys::CL_MEM_READ_WRITE;
            let host_ptr = ptr::null_mut();

            let mut errcode_ret = 0i32;

            let cl_mem = opencl_sys::clCreateBuffer(
                context.0, 
                flags, 
                size, 
                host_ptr, 
                &mut errcode_ret
            );

            let ret = opencl_sys::CLStatus::new(errcode_ret).expect("failed to convert i32 to CLStatus");

            match ret {
                opencl_sys::CLStatus::CL_SUCCESS => {
                    Ok(MemoryObject(cl_mem))
                },

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// Increments the memory object reference count.
    pub fn retain(&self) -> Result {

        unsafe {

            match opencl_sys::clRetainMemObject(self.0) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(()),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// Decrements the memory object reference count.
    pub fn release(&self) -> Result {

        unsafe {
            
            match opencl_sys::clReleaseMemObject(self.0) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(()),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }
}

impl Clone for MemoryObject {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        MemoryObject(self.0)
    }
}

impl Drop for MemoryObject {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}
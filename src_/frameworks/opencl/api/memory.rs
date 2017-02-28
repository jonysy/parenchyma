use std::ptr;
use super::error::Result;
use super::{Context, sys};

#[derive(Debug)]
pub struct Memory(pub(super) sys::cl_mem);

impl Memory {

    /// Creates a buffer object.
    pub fn create_buffer(context: &Context, size: usize) -> Result<Memory> {

        unsafe {
            // TODO https://streamcomputing.eu/blog/2013-02-03/opencl-basics-flags-for-the-creating-memory-objects/
            let flags = sys::CL_MEM_READ_WRITE;
            let host_ptr = ptr::null_mut();

            let mut errcode_ret = 0i32;

            let cl_mem = sys::clCreateBuffer(
                **context, 
                flags, 
                size, 
                host_ptr, 
                &mut errcode_ret
            );

            let ret = sys::CLStatus::new(errcode_ret)
                .expect("failed to convert i32 to CLStatus");

            result!(ret => Ok(Memory(cl_mem)))
        }
    }

    /// Increments the memory object reference count.
    pub fn retain(&self) -> Result {

        unsafe {

            result!(sys::clRetainMemObject(self.0))
        }
    }

    /// Decrements the memory object reference count.
    pub fn release(&self) -> Result {

        unsafe {
            
            result!(sys::clReleaseMemObject(self.0))
        }
    }
}

impl Clone for Memory {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        Memory(self.0)
    }
}

impl Drop for Memory {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}
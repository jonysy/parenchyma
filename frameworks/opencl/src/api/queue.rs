use opencl_sys;
use error::{ErrorKind, Result};
use super::{ContextPtr, DevicePtr};

#[derive(Debug)]
pub struct QueuePtr(opencl_sys::cl_command_queue);

impl QueuePtr {

    /// Create a command-queue on a specific device.
    pub fn new(context_ptr: &ContextPtr, device_ptr: &DevicePtr, properties: u64) -> Result<Self> {

        unsafe {
            let mut errcode_ret = 0i32;

            let cl_command_queue = opencl_sys::clCreateCommandQueue(
                context_ptr.0, 
                device_ptr.0, 
                properties, 
                &mut errcode_ret
            );

            let ret = opencl_sys::CLStatus::new(errcode_ret).expect("failed to convert i32 to CLStatus");

            match ret {
                opencl_sys::CLStatus::CL_SUCCESS => {
                    Ok(QueuePtr(cl_command_queue))
                },

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// Increments the command_queue reference count.
    pub fn retain(&self) -> Result {

        unsafe {
            match opencl_sys::clRetainCommandQueue(self.0) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(()),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// Decrements the command_queue reference count.
    pub fn release(&self) -> Result {

        unsafe {
            match opencl_sys::clReleaseCommandQueue(self.0) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(()),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }
}

impl Clone for QueuePtr {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        QueuePtr(self.0)
    }
}

impl Drop for QueuePtr {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}
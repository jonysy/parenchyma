use std::ops::Deref;
use super::error::Result;
use super::{Context, Device, sys};

#[derive(Debug)]
pub struct Queue(sys::cl_command_queue);

impl Queue {

    /// Create a command-queue on a specific device.
    pub fn new(context: &Context, device: &Device, properties: u64) -> Result<Self> {

        unsafe {
            
            let mut errcode_ret = 0i32;

            let cl_command_queue = sys::clCreateCommandQueue(
                **context, 
                **device, 
                properties, 
                &mut errcode_ret
            );

            let ret = sys::CLStatus::new(errcode_ret)
                .expect("failed to convert i32 to CLStatus");

            result!(ret => Ok(Queue(cl_command_queue)))
        }
    }

    /// Increments the command_queue reference count.
    pub fn retain(&self) -> Result {

        unsafe {

            result!(sys::clRetainCommandQueue(self.0))
        }
    }

    /// Decrements the command_queue reference count.
    pub fn release(&self) -> Result {

        unsafe {

            result!(sys::clReleaseCommandQueue(self.0))
        }
    }
}

impl Clone for Queue {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        Queue(self.0)
    }
}

impl Drop for Queue {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}

impl Deref for Queue {
    
    type Target = sys::cl_command_queue;
    
    fn deref(&self) -> &Self::Target {
        
        &self.0
    }
}
use std::{mem, ptr};
use std::ops::Deref;
use std::os::raw::c_void;
use super::Device;
use super::error::Result;
use super::sys;

#[derive(Debug)]
pub struct Context(sys::cl_context);

impl Context {
    /// Creates an OpenCL context.
    ///
    /// An OpenCL context is created with one or more devices. Contexts are used by the OpenCL 
    /// runtime for managing objects such as command-queues, memory, program and kernel objects 
    /// and for executing kernels on one or more devices specified in the context.
    pub fn new(devices: &[Device]) -> Result<Self> {

        unsafe {

            let mut errcode_ret = 0i32;

            let properties: *const isize = ptr::null();

            let num_devices = devices.len() as u32;

            let user_data: *mut c_void = ptr::null_mut();

            let callback: extern fn(*const i8, *const c_void, usize, *mut c_void) 
                = mem::transmute(ptr::null::<fn()>());

            let cl_context = sys::clCreateContext(
                properties, 
                num_devices, 
                devices.as_ptr() as *const *mut c_void, 
                callback, 
                user_data, 
                &mut errcode_ret
            );

            let ret = sys::CLStatus::new(errcode_ret).expect("failed to convert i32 to CLStatus");
            
            result!(ret => Ok(Context(cl_context)))
        }
    }

    /// Increment the context reference count.
    fn retain(&self) -> Result {

        unsafe {
            
            result!(sys::clRetainContext(self.0))
        }
    }

    /// Decrement the context reference count.
    fn release(&self) -> Result {

        unsafe {
            
            result!(sys::clReleaseContext(self.0))
        }
    }
}

impl Clone for Context {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        Context(self.0)
    }
}

impl Deref for Context {
    
    type Target = sys::cl_context;
    
    fn deref(&self) -> &Self::Target {
        
        &self.0
    }
}

impl Drop for Context {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}

impl From<sys::cl_context> for Context {
    
    fn from(cl_context: sys::cl_context) -> Self {
        
        Context(cl_context)
    }
}

impl Into<sys::cl_context> for Context {
    
    fn into(self) -> sys::cl_context {
        self.0
    }
}

impl PartialEq<Context> for Context {

    fn eq(&self, other: &Context) -> bool {

        self.0 == other.0
    }
}
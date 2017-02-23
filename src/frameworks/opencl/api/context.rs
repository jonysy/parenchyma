use std::{mem, ptr};
use std::ffi::CString;
use std::ops::Deref;
use std::os::raw::c_void;
use super::{Device, Program};
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
            let raw_devices: Vec<*mut c_void> = devices.iter().map(|d| *d.deref()).collect();

            let user_data: *mut c_void = ptr::null_mut();

            let callback: extern fn(*const i8, *const c_void, usize, *mut c_void) 
                = mem::transmute(ptr::null::<fn()>());

            let cl_context = sys::clCreateContext(
                properties, 
                num_devices, 
                raw_devices.as_ptr(), 
                callback, 
                user_data, 
                &mut errcode_ret
            );

            let ret = sys::CLStatus::new(errcode_ret).expect("failed to convert i32 to CLStatus");
            
            result!(ret => Ok(Context(cl_context)))
        }
    }

    /// Creates a program object for a context, and loads the source code specified by the text 
    /// strings in the strings array into the program object.
    pub fn create_program_with_source<I>(&self, strings: &[I]) -> Result<Program> 
        where I: AsRef<str> {

        unsafe {
            let mut errcode = 0i32;

            let n = strings.len() as u32;
            let lengths: Vec<usize> = strings.iter().map(|s| s.as_ref().len() as usize).collect();
            let lens_ptr = lengths.as_ptr();
            // https://doc.rust-lang.org/std/ffi/struct.CString.html#method.as_ptr
            let cstrings: Vec<CString> = strings.iter().map(|s| CString::new(s.as_ref()).unwrap()).collect();
            let ptrs: Vec<*const i8> = cstrings.iter().map(|s| s.as_ptr()).collect();
            let ptr = ptrs.as_ptr();

            let cl_program = sys::clCreateProgramWithSource(self.0, n, ptr, lens_ptr, &mut errcode);

            let ret = sys::CLStatus::new(errcode).expect("failed to convert i32 to CLStatus");

            result!(ret => Ok(Program::from(cl_program)))
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
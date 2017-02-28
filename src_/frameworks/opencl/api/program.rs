use std::{mem, ptr};
use std::ffi::CString;
use std::ops::Deref;
use super::{Device, Kernel};
use super::error::Result;
use super::sys;

#[derive(Debug)]
pub struct Program(sys::cl_program);

impl Program {

    /// Builds (compiles and links) a program executable from the program source or binary.
    pub fn build(&self, devices: &[Device]) -> Result {
        unsafe {
            let num_devices = devices.len() as u32;
            let raw_devices: Vec<*mut _> = devices.iter().map(|d| *d.deref()).collect();

            let options = ptr::null();
            let pfn_notify = mem::transmute(ptr::null::<fn()>());
            let user_data = ptr::null_mut();

            let ret = sys::clBuildProgram(self.0, num_devices, raw_devices.as_ptr(),
                options, pfn_notify, user_data);

            result!(ret)
        }
    }

    /// Creates a kernal object.
    pub fn create_kernel<T>(&self, name: T) -> Result<Kernel> where T: AsRef<str> {
        unsafe {
            let mut errcode = 0i32;
            let cstring = CString::new(name.as_ref()).unwrap();
            let ptr = cstring.as_ptr();
            let cl_kernel = sys::clCreateKernel(self.0, ptr, &mut errcode);
            let ret = sys::CLStatus::new(errcode).expect("failed to convert i32 to CLStatus");
            result!(ret => Ok(Kernel::from(cl_kernel)))
        }
    }

    /// Increment the context reference count.
    fn retain(&self) -> Result {

        unsafe {
            
            result!(sys::clRetainProgram(self.0))
        }
    }

    /// Decrement the context reference count.
    fn release(&self) -> Result {

        unsafe {
            
            result!(sys::clReleaseProgram(self.0))
        }
    }
}

impl Clone for Program {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        Program(self.0)
    }
}

impl Drop for Program {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}

impl From<sys::cl_program> for Program {
    
    fn from(cl_program: sys::cl_program) -> Self {
        
        Program(cl_program)
    }
}
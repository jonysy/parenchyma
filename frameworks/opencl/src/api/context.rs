use opencl_sys;
use std::os::raw::c_void;
use std::{mem, ptr};

use error::{ErrorKind, Result};
use super::DevicePtr;

#[derive(Debug)]
pub struct ContextPtr(opencl_sys::cl_context);

impl ContextPtr {

    // clCreateContext(
    //     properties: *const cl_context_properties,
    //     num_devices: cl_uint,
    //     devices: *const cl_device_id,
    //     pfn_notify: extern fn (*const libc::c_char, *const raw::c_void, libc::size_t, *mut raw::c_void),
    //     user_data: *mut raw::c_void,
    //     errcode_ret: *mut cl_int
    // ) -> cl_context;

    /// Creates an OpenCL context.
    ///
    /// An OpenCL context is created with one or more devices. Contexts are used by the OpenCL 
    /// runtime for managing objects such as command-queues, memory, program and kernel objects 
    /// and for executing kernels on one or more devices specified in the context.
    pub fn new(
        device_ptrs: &[DevicePtr],
    ) -> Result<Self> {

        unsafe {

            let mut errcode_ret = 0i32;

            let properties: *const isize = ptr::null();

            let num_devices = device_ptrs.len() as u32;

            let user_data: *mut c_void = ptr::null_mut();

            let device_raw_ptrs: Vec<opencl_sys::cl_device_id> = device_ptrs.iter().map(|d| d.0).collect();

            let callback: extern fn(*const i8, *const c_void, usize, *mut c_void) 
                = mem::transmute(ptr::null::<fn()>());

            let cl_context = opencl_sys::clCreateContext(
                properties, 
                num_devices, 
                device_raw_ptrs.as_ptr(), 
                callback, 
                user_data, 
                &mut errcode_ret
            );

            let ret = opencl_sys::CLStatus::new(errcode_ret).expect("failed to convert i32 to CLStatus");

            match ret {
                opencl_sys::CLStatus::CL_SUCCESS => {
                    Ok(ContextPtr(cl_context))
                },

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// Increment the context reference count.
    pub fn retain(&self) -> Result {

        unsafe {
            match opencl_sys::clRetainContext(self.0) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(()),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// Decrement the context reference count.
    pub fn release(&self) -> Result {

        unsafe {
            match opencl_sys::clReleaseContext(self.0) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(()),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }
}

impl Clone for ContextPtr {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        ContextPtr(self.0)
    }
}

impl Drop for ContextPtr {

    fn drop(&mut self) {

        if let Err(e) = self.release() {
            panic!("{}", e);
        }
    }
}
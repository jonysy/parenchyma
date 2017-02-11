use opencl_sys;
use std::ptr;

use error::{ErrorKind, Result};
use super::DevicePtr;

#[derive(Debug)]
pub struct PlatformPtr(pub(super) opencl_sys::cl_platform_id);

  // pub fn clGetPlatformInfo(platform: cl_platform_id,
  //                          param_name: cl_platform_info,
  //                          param_value_size: libc::size_t,
  //                          param_value: *mut libc::c_void,
  //                          param_value_size_ret: *mut libc::size_t) -> cl_int;

impl PlatformPtr {

    fn info_size(&self, p: u32) -> Result<usize> {

        unsafe {

            let mut size = 0;

            match opencl_sys::clGetPlatformInfo(self.0, p, 0, ptr::null_mut(), &mut size) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(size),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    fn info(&self, p: u32) -> Result<String> {

        unsafe {

            let size = self.info_size(p)?;

            let mut b = vec![0u8; size];

            let null = ptr::null_mut();

            match opencl_sys::clGetPlatformInfo(self.0, p, size, b.as_mut_ptr() as *mut _, null) {
                opencl_sys::CLStatus::CL_SUCCESS => {

                    Ok(String::from_utf8(b).unwrap())
                },

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    /// OpenCL profile string. Returns the profile name supported by the implementation. The 
    /// profile name returned can be one of the following strings:
    ///
    /// FULL_PROFILE - if the implementation supports the OpenCL specification (functionality 
    ///                defined as part of the core specification and does not require any 
    ///                extensions to be supported).
    ///
    /// EMBEDDED_PROFILE - if the implementation supports the OpenCL embedded profile. The embedded 
    ///                    profile is defined to be a subset for each version of OpenCL.
    pub fn profile(&self) -> Result<String> {

        self.info(opencl_sys::CL_PLATFORM_PROFILE)
    }

    /// Platform name
    pub fn name(&self) -> Result<String> {

        self.info(opencl_sys::CL_PLATFORM_NAME)
    }

    /// Platform vendor
    pub fn vendor(&self) -> Result<String> {

        self.info(opencl_sys::CL_PLATFORM_VENDOR)
    }

    /// Returns a space-separated list of extension names (the extension names themselves do 
    /// not contain any spaces) supported by the platform. Extensions defined here must be 
    /// supported by all devices associated with this platform.
    pub fn extensions(&self) -> Result<Vec<String>> {

        self.info(opencl_sys::CL_PLATFORM_EXTENSIONS).map(|st| {
            st.split_whitespace().map(|s| s.into()).collect()
        })
    }

    pub fn ndevices_by_type(&self, t: u64) -> Result<u32> {

        unsafe {

            let mut ndevices = 0;

            match opencl_sys::clGetDeviceIDs(self.0, t, 0, ptr::null_mut(), &mut ndevices) {
                opencl_sys::CLStatus::CL_SUCCESS => Ok(ndevices),

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    pub fn devices_by_type(&self, t: u64) -> Result<Vec<DevicePtr>> {

        unsafe {

            let ndevices = self.ndevices_by_type(t)?;
            let mut vec_id = vec![0 as opencl_sys::cl_device_id; ndevices as usize];
            let n = ptr::null_mut();

            match opencl_sys::clGetDeviceIDs(self.0, t, ndevices, vec_id.as_mut_ptr(), n) {
                opencl_sys::CLStatus::CL_SUCCESS => {
                    Ok(vec_id.iter().map(|&id| DevicePtr(id)).collect())
                },

                e @ _ => Err((e.into(): ErrorKind).into())
            }
        }
    }

    pub fn all_device_ids(&self) -> Result<Vec<DevicePtr>> {

        self.devices_by_type(opencl_sys::CL_DEVICE_TYPE_ALL)
    }
}
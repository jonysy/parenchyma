use std::ptr;

use super::error::Result;
use super::Device;
use super::sys;

#[derive(Debug)]
pub struct Platform(pub(super) sys::cl_platform_id);

impl Platform {

    fn info_size(&self, p: u32) -> Result<usize> {

        unsafe {

            let mut size = 0;

            result!(
                sys::clGetPlatformInfo(self.0, p, 0, ptr::null_mut(), &mut size) => Ok(size))
        }
    }

    fn info(&self, p: u32) -> Result<String> {

        unsafe {

            let size = self.info_size(p)?;

            let mut b = vec![0u8; size];

            let null = ptr::null_mut();

            result!(
                sys::clGetPlatformInfo(self.0, p, size, b.as_mut_ptr() as *mut _, null) 
                => Ok(String::from_utf8(b).unwrap()))
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

        self.info(sys::CL_PLATFORM_PROFILE)
    }

    /// Platform name
    pub fn name(&self) -> Result<String> {

        self.info(sys::CL_PLATFORM_NAME)
    }

    /// Platform vendor
    pub fn vendor(&self) -> Result<String> {

        self.info(sys::CL_PLATFORM_VENDOR)
    }

    /// Returns a space-separated list of extension names (the extension names themselves do 
    /// not contain any spaces) supported by the platform. Extensions defined here must be 
    /// supported by all devices associated with this platform.
    pub fn extensions(&self) -> Result<Vec<String>> {

        self.info(sys::CL_PLATFORM_EXTENSIONS).map(|st| {
            st.split_whitespace().map(|s| s.into()).collect()
        })
    }

    pub fn ndevices_by_type(&self, t: u64) -> Result<u32> {

        unsafe {

            let mut ndevices = 0;

            result!(
                sys::clGetDeviceIDs(self.0, t, 0, ptr::null_mut(), &mut ndevices)
                => Ok(ndevices))
        }
    }

    pub fn devices_by_type(&self, t: u64) -> Result<Vec<Device>> {

        unsafe {

            let ndevices = self.ndevices_by_type(t)?;
            let mut vec_id = vec![0 as sys::cl_device_id; ndevices as usize];
            let n = ptr::null_mut();

            result!(
                sys::clGetDeviceIDs(self.0, t, ndevices, vec_id.as_mut_ptr(), n)
                => Ok(vec_id.iter().map(|&id| Device::from(id)).collect()))
        }
    }

    pub fn all_device_ids(&self) -> Result<Vec<Device>> {

        self.devices_by_type(sys::CL_DEVICE_TYPE_ALL)
    }
}
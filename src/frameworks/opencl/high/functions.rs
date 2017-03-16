use std::ptr;

use super::{Platform, Result};
use super::utility;
use super::super::foreign;

/// Number of platforms
pub fn nplatforms() -> Result<u32> {
    unsafe {
        let mut nplatforms = 0;
        let ret_value = foreign::clGetPlatformIDs(0, ptr::null_mut(), &mut nplatforms);
        return utility::check(ret_value, || nplatforms);
    }
}

/// Obtain the list of platforms available.
pub fn platforms() -> Result<Vec<Platform>> {
    unsafe {
        let nplatforms = nplatforms()?;
        let mut vec_id = vec![ptr::null_mut(); nplatforms as usize];
        let pointer = vec_id.as_mut_ptr();
        let ret_value = foreign::clGetPlatformIDs(nplatforms, pointer, ptr::null_mut());
        return utility::check(ret_value, || vec_id.iter().map(|&id| Platform(id)).collect());
    }
}
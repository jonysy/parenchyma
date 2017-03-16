use super::{ErrorKind, Result};
use super::super::foreign::CLStatus;

pub fn check<F, T>(cl_status: CLStatus, ok_value: F) -> Result<T> where F: FnOnce() -> T {
    
    match cl_status {
        CLStatus::CL_SUCCESS => Ok(ok_value()),

        e @ _ => {
            let kind: ErrorKind = e.into();
            let error = kind.into();
            Err(error)
        }
    }
}
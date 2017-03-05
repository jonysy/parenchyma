use super::error::{ErrorKind, Result};
use super::super::cl;

pub fn check<F, T>(cl_status: cl::CLStatus, ok_value: F) -> Result<T> where F: FnOnce() -> T {

    match cl_status {
        cl::CLStatus::CL_SUCCESS => Ok(ok_value()),

        e @ _ => {
            let kind: ErrorKind = e.into();
            let error = kind.into();
            Err(error)
        }
    }
}
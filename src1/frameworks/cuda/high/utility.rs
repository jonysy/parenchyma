use std::error;
use super::error::{Error, ErrorKind, Result};
use super::super::sh::cudaError_enum;

pub fn check<F, T>(cl_status: cudaError_enum, ok_value: F) -> Result<T> where F: FnOnce() -> T {
    
    match cl_status {
        cudaError_enum::CUDA_SUCCESS => Ok(ok_value()),

        e @ _ => {
            let kind: ErrorKind = e.into();
            let error = kind.into();
            Err(error)
        }
    }
}

pub fn check_with<F, E, T>(cl_status: cudaError_enum, error: E, ok_value: F) -> Result<T> 
    where E: Into<Box<error::Error + Send + Sync>>,
          F: FnOnce() -> T {
    
    match cl_status {
        cudaError_enum::CUDA_SUCCESS => Ok(ok_value()),

        e @ _ => {
            Err(Error::new(e, error))
        }
    }
}
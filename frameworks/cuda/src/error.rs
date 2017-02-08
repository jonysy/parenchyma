use cuda;
use std::{error, fmt};

#[derive(Debug)]
pub struct CudaError;

impl fmt::Display for CudaError {

    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {

        unimplemented!()
    }
}

impl error::Error for CudaError {

    fn description(&self) -> &str {

        unimplemented!()
    }
}

impl From<cuda::error::Error> for CudaError {

    fn from(e: cuda::error::Error) -> CudaError {

        unimplemented!()
    }
}
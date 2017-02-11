use opencl;
use std::{error, fmt, result};

pub type Result<T = ()> = result::Result<T, OpenCLError>;

#[derive(Debug)]
pub struct OpenCLError;

impl fmt::Display for OpenCLError {

    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {

        panic!("write!")
    }
}

impl error::Error for OpenCLError {

    fn description(&self) -> &str {

        unimplemented!()
    }
}

impl From<opencl::error::Error> for OpenCLError {

    fn from(e: opencl::error::Error) -> OpenCLError {

        panic!("error!")
    }
}
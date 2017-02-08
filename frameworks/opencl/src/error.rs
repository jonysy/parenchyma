use std::{error, fmt};

#[derive(Debug)]
pub struct OpenCLError;

impl fmt::Display for OpenCLError {

    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {

        unimplemented!()
    }
}

impl error::Error for OpenCLError {

    fn description(&self) -> &str {

        unimplemented!()
    }
}
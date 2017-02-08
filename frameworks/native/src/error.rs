use parenchyma::error::{Error, FrameworkSpecificError};
use std::{error, fmt};
use super::Native;

#[derive(Debug)]
pub struct NativeError(Error);

impl From<Error> for NativeError {

    fn from(error: Error) -> Self {
        NativeError(error)
    }
}

impl fmt::Display for NativeError {

    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {

        self.0.fmt(fmt)
    }
}

impl error::Error for NativeError {

    fn description(&self) -> &str {

        self.0.description()
    }

    fn cause(&self) -> Option<&error::Error> {

        self.0.cause()
    }
}

impl FrameworkSpecificError for NativeError {
    type F = Native;
}
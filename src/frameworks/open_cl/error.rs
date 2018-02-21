use ocl::Error as OpenCLError;
use error::{Error, ErrorKind};

impl From<OpenCLError> for Error {
    /// Creates a new error from a known kind of error
    fn from(e: OpenCLError) -> Error {
        Error::new(ErrorKind::Framework(super::OpenCL::ID), ::std::error::Error::description(&e)) 
    }
}
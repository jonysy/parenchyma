use super::{OpenCLPlatform, Result};
use super::api;
use super::super::super::{Error, ErrorKind};

/// Provides the OpenCL framework.
///
/// ```rust
/// extern crate parenchyma;
/// extern crate parenchyma_opencl;
///
/// use parenchyma::{Backend, Framework};
/// use parenchyma_opencl::OpenCL;
// --- work around: https://github.com/rust-lang/cargo/issues/960
///
/// # fn main() {
/// let framework = OpenCL::new().expect("failed to construct framework");
/// let selection = framework.available_platforms[0].available_devices.clone();
/// let backend = Backend::new(framework, selection).expect("failed to construct backend");
/// # }
/// ```
#[derive(Debug)]
pub struct OpenCL {
    /// List of available platforms.
    pub available_platforms: Vec<OpenCLPlatform>,
}

impl OpenCL {
    /// The name of the framework.
    const FRAMEWORK_NAME: &'static str = "OPEN_CL";
    
    /// Initializes the framework.
    pub fn try_new() -> Result<Self> {
        let mut platform_ptrs = api::platform_ids()?;
        let capacity = platform_ptrs.len();
        let mut available_platforms = Vec::with_capacity(capacity);
        
        for _ in 0..capacity {

            let ptr = platform_ptrs.remove(0);
            let p = OpenCLPlatform::new(ptr)?;
            available_platforms.push(p)
        }

        Ok(OpenCL { available_platforms: available_platforms })
    }
}

impl From<api::error::Error> for Error {

    fn from(e: api::error::Error) -> Self {

        Error::new(ErrorKind::Framework { name: OpenCL::FRAMEWORK_NAME }, e)
    }
}
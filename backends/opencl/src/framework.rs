use cl;
use parenchyma::Framework;
use super::{OpenCLContext, OpenCLDevice, OpenCLError, OpenCLMemory, OpenCLPlatform};

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
/// let framework = OpenCL::new();
/// let selection = framework.available_platforms[0].available_devices.clone();
/// let backend = Backend::new(framework, selection).expect("failed to construct backend");
/// # }
/// ```
#[derive(Debug)]
pub struct OpenCL {
    /// List of available platforms.
    pub available_platforms: Vec<OpenCLPlatform>,
}

impl Framework for OpenCL {
    /// The name of the framework.
    const FRAMEWORK_NAME: &'static str = "OPEN_CL";

    /// The context representation.
    type Context = OpenCLContext;

    /// The device representation.
    type D = OpenCLDevice;

    /// An error type associated with the framework.
    type E = OpenCLError;

    /// The memory representation.
    ///
    /// Memory objects are OpenCL data that can be moved on and off devices and can be classified as:
    ///
    /// * Buffers - Contiguous chunks of memory (arrays, pointers, structs). read/write capable
    /// * Images - Opaque 2D or 3D objects. Can either read or written in a kernel, but not both
    type M = OpenCLMemory;

    /// Initializes the framework.
    fn new() -> Result<Self, OpenCLError> {
        let available_platforms = cl::Platform::list().iter().map(From::from).collect();

        Ok(OpenCL { available_platforms: available_platforms })
    }

    /// Returns a default selection of devices for the framework.
    fn default_selection(&self) -> Vec<Self::D> {
        self.available_platforms[0].available_devices.clone()
    }
}
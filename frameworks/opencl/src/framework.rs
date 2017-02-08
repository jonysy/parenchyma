use parenchyma::Framework;
use super::cl;
use super::{Context, Device, Memory, Platform};

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
    pub available_platforms: Vec<Platform>,
}

impl Framework for OpenCL {
    /// The name of the framework.
    const FRAMEWORK_NAME: &'static str = "OPEN_CL";

    /// The context representation.
    type Context = Context;

    /// The device representation.
    type Device = Device;

    /// The memory representation.
    type Memory = Memory;

    /// Initializes the framework.
    fn new() -> OpenCL {
        let available_platforms = cl::Platform::list().iter().map(From::from).collect();

        OpenCL {
            available_platforms: available_platforms,
        }
    }
}
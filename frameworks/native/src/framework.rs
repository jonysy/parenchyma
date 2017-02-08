use parenchyma::{Framework, Processor};
use parenchyma::error::{Error, Result};
use super::{NativeContext, NativeDevice, NativeMemory};

/// Provides the native framework.
///
/// ```rust
/// extern crate parenchyma;
/// extern crate parenchyma_native;
///
/// use parenchyma::{Backend, Framework};
/// use parenchyma_native::Native;
// --- work around: https://github.com/rust-lang/cargo/issues/960
///
/// # fn main() {
/// let framework = Native::new();
/// let selection = framework.available_devices.clone();
/// let backend = Backend::new(framework, selection).expect("failed to construct backend");
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Native {
    pub available_devices: Vec<NativeDevice>,
}

impl Framework for Native {
    const FRAMEWORK_NAME: &'static str = "NATIVE";

    type Context = NativeContext;

    type D = NativeDevice;

    type E = Error;

    type M = NativeMemory;

    fn new() -> Result<Self> {
        Ok(Native { 
            available_devices: vec![NativeDevice {
                name: "Host CPU",
                compute_units: 1,
                processor: Processor::Cpu,
            }]
        })
    }

    fn default_selection(&self) -> Vec<NativeDevice> {
        self.available_devices.clone()
    }
}
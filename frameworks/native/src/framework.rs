use parenchyma::{Framework, Processor};
use std::borrow::Cow;
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

    /// The context representation.
    type Context = NativeContext;

    /// The device representation.
    type Device = NativeDevice;

    /// The memory representation.
    type Memory = NativeMemory;

    /// Initializes a the framework.
    fn new() -> Native {
        let device = NativeDevice {
            name: Cow::from("Host CPU"),
            compute_units: 1,
            processor: Processor::Cpu,
        };

        Native { available_devices: vec![device] }
    }
}

impl Default for ::parenchyma::Backend<Native> {

    fn default() -> Self {

        unimplemented!()
    }
}
use super::super::{Framework, DeviceKind};
use super::super::error::{Error, Result};
use super::{NativeContext, NativeDevice, NativeMemory};

/// The `Native` framework simply represents the host CPU.
///
/// ```rust
/// use parenchyma::{Backend, Framework, Native};
///
/// let framework = Native::new().expect("failed to initialize framework");
/// let selection = framework.available_devices.clone();
/// let backend = Backend::new(framework, selection).expect("failed to construct backend");
/// ```
#[derive(Clone, Debug)]
pub struct Native {
    /// Available devices
    pub available_devices: Vec<NativeDevice>,
}

impl Framework for Native {
    const FRAMEWORK_NAME: &'static str = "NATIVE";

    type Context = NativeContext;

    type Device = NativeDevice;

    type E = Error;

    type M = NativeMemory;

    fn new() -> Result<Self> {
        Ok(Native { 
            available_devices: vec![NativeDevice {
                name: "Host CPU",
                compute_units: 1,
                kind: DeviceKind::Cpu,
            }]
        })
    }

    fn default_selection(&self) -> Vec<NativeDevice> {
        self.available_devices.clone()
    }
}
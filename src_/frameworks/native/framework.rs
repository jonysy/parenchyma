use super::NativeDevice;
use super::super::super::DeviceKind;

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

impl Native {

    // const FRAMEWORK_NAME: &'static str = "NATIVE";

    pub fn new() -> Self {
        Native { 
            available_devices: vec![NativeDevice {
                name: "Host CPU",
                compute_units: 1,
                kind: DeviceKind::Cpu,
            }]
        }
    }
}

// impl Framework for Native {
    
//     const FRAMEWORK_NAME: &'static str = "NATIVE";

//     type Err = !;
// }
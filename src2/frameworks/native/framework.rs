use framework::{Framework, FrameworkExt};
use hardware::{Hardware, HardwareType};
use super::{HARDWARE, NativeContext};

/// The native framework
///
/// # Example
///
/// ```rust
/// use parenchyma::prelude::*;
///
/// let framework = Native;
/// let selection = framework.hardware().to_vec();
/// let ref host: Backend = Backend::new::<Native>().unwrap();
///
/// let sh: SharedTensor = SharedTensor::with(host, [2, 2], vec![1., 2., 3., 4.]).unwrap();
///
/// let tensor = sh.read(host).unwrap();
///
/// println!("{:#?}", tensor);
/// ```
#[derive(Debug, Default)]
pub struct Native;

impl Framework for Native {
    const ID: &'static str = "NATIVE";

    fn hardware(&self) -> &[Hardware] {
        &HARDWARE
    }
}

impl FrameworkExt for Native { type Context = NativeContext; }
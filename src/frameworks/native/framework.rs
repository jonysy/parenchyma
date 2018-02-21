use super::NativeContext;
use super::super::super::error::Result;
use super::super::super::framework::{Framework, FrameworkCtor};
use super::super::super::hardware::{Hardware, HardwareKind};

/// The native framework
#[derive(Debug)]
pub struct Native {
    hardware: [Hardware; 1],
}

impl Native {
    const ID: &'static str = "native/host";
}

impl Framework for Native {
    fn name(&self) -> &'static str {
        return Native::ID;
    }

    fn hardware(&self) -> &[Hardware] {
        &self.hardware
    }
}

impl<P> FrameworkCtor<P> for Native where P: 'static {
    type Context = NativeContext<P>;

    fn new() -> Result<Self> {
        Ok(Native {
            hardware: [Hardware {
                id: 0usize,
                framework: Native::ID,
                kind: HardwareKind::CPU,
                name: String::from("Host CPU"),
                compute_units: 1,
            }],
        })
    }
}
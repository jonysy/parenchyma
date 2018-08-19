use super::NativeContext;
use super::super::super::error::Result;
use super::super::super::framework::{Framework, FrameworkCtor};
use super::super::super::hardware::{Hardware, HardwareKind};

use std::marker::PhantomData;

/// The native framework
#[derive(Debug)]
pub struct Native<P> {
    hardware: [Hardware; 1],
    package: PhantomData<P>,
}

impl<P> Native<P> {
    const ID: &'static str = "native/host";
}

impl<P> Framework for Native<P> where P: 'static {
    fn name(&self) -> &'static str {
        return Native::<P>::ID;
    }

    fn hardware(&self) -> &[Hardware] {
        &self.hardware
    }
}

impl<P> FrameworkCtor for Native<P> where P: 'static {
    type Context = NativeContext<P>;

    fn new() -> Result<Self> {
        Ok(Native {
            hardware: [Hardware {
                id: 0usize,
                framework: Native::<P>::ID,
                kind: HardwareKind::CPU,
                name: String::from("Host CPU"),
                compute_units: 1,
            }],
            package: PhantomData,
        })
    }
}
use context::{Context, ContextConfig};
use error::Result;
use hardware::{Hardware, HardwareDevice};
use super::{HARDWARE, HOST, Native};

#[derive(Debug)]
pub struct NativeContext;

impl Context for NativeContext {
    /// Returns all _activatable_ hardware provided to the context.
    fn selection(&self) -> &[Hardware] {
        &HARDWARE
    }

    /// Returns the _active_ device.
    fn active(&self) -> &(HardwareDevice + 'static) {
        &HOST
    }

    fn activate(&mut self, _: usize) -> Result {
        Ok(())
    }
}

impl<'a> From<ContextConfig<'a, Native>> for NativeContext {

    fn from(_: ContextConfig<'a, Native>) -> NativeContext {
        NativeContext
    }
}
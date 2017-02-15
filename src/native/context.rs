use super::super::Context;
use super::super::error::Result;
use super::{Native, NativeDevice};

/// Native context
#[derive(Clone, Debug)]
pub struct NativeContext {
    /// Selected devices
    selected_devices: Vec<NativeDevice>,
}

impl Context for NativeContext {
    type F = Native;

    fn new(devices: Vec<NativeDevice>) -> Result<Self> {
        
        Ok(NativeContext { selected_devices: devices })
    }

    fn devices(&self) -> &[NativeDevice] {
        &self.selected_devices
    }
}
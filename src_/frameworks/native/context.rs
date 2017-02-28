use super::NativeDevice;

/// Native context
#[derive(Clone, Debug)]
pub struct NativeContext {
    /// Selected devices
    selected_devices: Vec<NativeDevice>,
}

impl NativeContext {

    /// Constructs a context from a selection of devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - a list of devices.
    pub fn new(devices: Vec<NativeDevice>) -> Self {
        
        NativeContext { selected_devices: devices }
    }

    pub fn devices(&self) -> &[NativeDevice] {
        &self.selected_devices
    }
}
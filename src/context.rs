use std::fmt::Debug;
use super::{Device, ExtensionPackage, Result};
use utility::Has;

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
pub trait Context: Debug {

    /// The extension package built for `Self`.
    type Package: ExtensionPackage;

    /// Returns the _active_ device.
    fn active_device(&self) -> &Device;

    /// Set the device at the specified `index` as the active device.
    fn set_active(&mut self, index: usize) -> Result;

    #[doc(hidden)]
    fn extension(&self) -> &<Self::Package as ExtensionPackage>::Extension;
}

impl<I> Has<Device> for I where I: Context {

    fn get_ref(&self) -> &Device {
        self.active_device()
    }
}
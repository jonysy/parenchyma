use {Context, Device, ExtensionPackage, Result};
use std::marker::{PhantomData, Unsize};
use super::NativeDevice;

/// The native context.
#[derive(Clone, Debug)]
pub struct NativeContext<X>(pub PhantomData<X>);

impl<X> Context for NativeContext<X> where X: ExtensionPackage, Self: Unsize<X::Extension> {

    type Package = X;

    fn active_device(&self) -> &Device {
        static NATIVE_DEVICE: NativeDevice = NativeDevice;
        &NATIVE_DEVICE
    }

    fn set_active(&mut self, _: usize) -> Result { Ok(()) }

    fn extension(&self) -> &X::Extension {
        self
    }
}
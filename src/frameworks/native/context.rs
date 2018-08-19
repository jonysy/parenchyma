use std::marker::PhantomData;

use std::marker::Unsize;
use super::Native;
use super::super::super::compute_device::ComputeDevice;
use super::super::super::context::{Context, ContextCtor};
use super::super::super::error::Result;
use super::super::super::extension_package::ExtensionPackage;
use super::super::super::hardware::Hardware;

/// Defines a Native context.
pub struct NativeContext<P = ()>(PhantomData<P>);

impl<Package> Context for NativeContext<Package> 
    where Package: ExtensionPackage, 
          NativeContext<Package>: Unsize<Package::Extension> {

    type Package = Package;

    fn active_codev(&self) -> &ComputeDevice {
        &super::HOST
    }

    fn activate(&mut self, _: usize) -> Result {
        Ok(())
    }

    fn extension(&self) -> &<Package as ExtensionPackage>::Extension {
        self
    }
}

impl<P> ContextCtor<P> for NativeContext<P>
    where P: 'static + ExtensionPackage, 
          NativeContext<P>: Unsize<P::Extension> {
            
    type F = Native<P>;

    fn new(_: &Self::F, _: &[Hardware]) -> Result<Self> {
        Ok(NativeContext(PhantomData))
    }
}
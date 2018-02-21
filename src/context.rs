//! Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
//! objects such as memory, command-queues, programs/modules and kernels.
//!
//! You can create a context encapsulating a selection of hardware via a [`Backend`].
//!
//! [`Backend`]: ./struct.Backend.html

use super::compute_device::ComputeDevice;
use super::error::Result;
use super::extension_package::ExtensionPackage;
use super::hardware::Hardware;

/// A trait implemented by all contexts.
pub trait Context: 'static {
    /// The extension package built for the framework's context.
    type Package: ExtensionPackage;
    /// Returns the active device.
    fn active_codev(&self) -> &ComputeDevice;
    /// Returns the package extension.
    fn extension(&self) -> &<Self::Package as ExtensionPackage>::Extension;
    // /// Returns all _activatable_ hardware provided to the context.
    // fn selection(&self) -> &[Hardware];
}

/// The non-object-safe part of the `Context`.
///
/// todo: generic associated types may help here..
pub trait ContextCtor<Package>
    where Self: Context<Package=Package> + Sized, 
          Package: ExtensionPackage {
    /// The framework representation for the context.
    type F;
    /// Constructs a new context from the `framework` and the `selection` of hardware.
    fn new(framework: &Self::F, selection: &[Hardware]) -> Result<Self>;
}
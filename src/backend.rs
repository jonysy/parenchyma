use std::ops;
use super::{BoxContext, Context, Device, Error, ExtensionPackage, Framework, Hardware, Unextended};
use super::Result;
use utility::{self, TryDefault};

/// The heart of Parenchyma - provides an interface for running parallel computations on one or 
/// more devices.
///
/// The `Backend` type is an abstraction over a [framework](./trait.Framework.html) and is used as 
/// a way to interact with your devices. A backend provides you with the functionality of managing 
/// the memory of the devices and copying memory objects to/from the host. Additionally, backends 
/// allow you to execute operations in parallel through kernel functions on devices.
///
/// ## Architecture
///
/// Backends are initialized by providing a framework and a selection of devices compatible with 
/// the framework to the [`Backend::new`](#method.new) associated function, or by simply 
/// calling [`Backend::default`](#method.default). The framework determines which devices are 
/// available and how parallel kernel functions can be executed.
#[derive(Debug)]
pub struct Backend<X = Unextended> {
    /// The initialized framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be used and
    /// determines which hardwares will be available and how parallel kernel functions can be
    /// executed.
    framework: Box<Framework>,
    /// The context associated with the `framework`.
    ///
    /// Contexts are the heart of both OpenCL and CUDA applications. Contexts are created from one 
    /// or more devices that are capable of executing methods and synchronizing memory. See 
    /// the [`Context`] trait for more information.
    ///
    /// [`Context`]: (./trait.Context.html)
    context: Box<Context<Package = X>>,
}

impl<X> Backend<X> where X: ExtensionPackage {

    /// Initialize a new backend.
    pub fn new<F>() -> Result<Self> where F: BoxContext<X> + Framework + TryDefault<Err = Error> {

        let framework = Box::new(F::try_default()?);
        let selection = framework.available_hardware();
        let context = framework.enclose(selection)?;

        Ok(Backend { framework: framework, context })
    }

    /// Constructs a backend from the specified `framework` and `selection`.
    pub fn with<F>(fwrk: F, selection: Vec<Hardware>) -> Result<Self> 
        where F: BoxContext<X> + Framework {

        let framework = Box::new(fwrk);
        let context = framework.enclose(selection)?;

        Ok(Backend { framework, context })
    }

    /// Set the device at the specified `index` as the active device.
    ///
    /// Only one device can be the _active_ device - the device in which operations are executed.
    pub fn set_active(&mut self, index: usize) -> Result {

        self.context.set_active(index)
    }
}

impl<X> ops::Deref for Backend<X> where X: ExtensionPackage {

    type Target = X::Extension;

    fn deref<'a>(&'a self) -> &'a X::Extension {

        self.context.extension()
    }
}

impl<X> utility::Has<Device> for Backend<X> where X: ExtensionPackage {

    fn get_ref(&self) -> &Device {
        self.context.active_device()
    }
}

// pub trait AsBackend { }
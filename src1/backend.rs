use std::ops;
use std::marker::PhantomData;
use super::{BoxContext, Context, Device, ExtensionPackage, Framework, Hardware, HardwareKind, Unextended};
use super::{Error, ErrorKind, Result};
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

        let framework = F::try_default()?;
        let selection = framework.available_hardware();
        Self::with(framework, selection)
    }

    /// Constructs a backend from the specified `framework` and `selection`.
    pub fn with<F>(fwrk: F, selection: Vec<Hardware>) -> Result<Self> 
        where F: BoxContext<X> + Framework {

        let mut framework = Box::new(fwrk);
        let context = framework.enclose(selection)?;

        Ok(Backend { framework, context })
    }

    /// Initialize a new Backend from a BackendConfig
    pub fn try_from<F>(co: BackendConfig<F, X>) -> Result<Self> where F: BoxContext<X> + Framework {

        let BackendConfig { framework, hardware, kind, .. } = co;
        let mut backend = Self::with(framework, hardware)?;
        backend.select(|h| h.kind == kind)?;

        Ok(backend)
    }

    /// Set the device at the specified `index` as the active device.
    ///
    /// Only one device can be the _active_ device - the device in which operations are executed.
    pub fn set_active(&mut self, index: usize) -> Result {

        self.context.set_active(index)
    }

    /// Select the first device that meets the specified requirements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parenchyma::{Backend, HardwareKind, Native};
    ///
    /// let mut native: Backend = Backend::new::<Native>().unwrap();
    /// assert!(native.select(|hardware| hardware.kind == HardwareKind::CPU).is_ok());
    /// ```
    pub fn select<P>(&mut self, mut predicate: P) -> Result where P: FnMut(&Hardware) -> bool {
        let opt_index = self.framework.selection()
            .iter()
            .enumerate()
            .filter(|&(_, hardware)| predicate(hardware))
            .map(|(index, _)| index)
            .nth(0);

        match opt_index {
            Some(index) => self.set_active(index),
            _ => {
                let message = "There are no devices matching the specified criteria.";
                Err(Error::new(ErrorKind::Framework { name: self.framework.name() }, message))
            }
        }
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

/// Provides Backend Configuration, used to initialize a new Backend.
#[derive(Debug)]
pub struct BackendConfig<F, X> {
    framework: F,
    hardware: Vec<Hardware>,
    kind: HardwareKind,
    extension: PhantomData<X>,
}

impl<F, X> BackendConfig<F, X> {

    /// Creates a new `BackendConfig`.
    pub fn new(framework: F, hardware: Vec<Hardware>, kind: HardwareKind) -> Self {

        BackendConfig { framework, hardware, kind, extension: PhantomData }
    }
}
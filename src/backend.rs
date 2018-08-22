//! The `Backend` is the heart of Parenchyma. It provides an interface for running parallel 
//! computations on one or more devices. It is the main and highest struct of Parenchyma.
//!
//! The `Backend` type is an abstraction over a [framework](./trait.Framework.html) and is used as 
//! a way to interact with your devices. You can create a backend for computation by first choosing 
//! a specific [framework](./trait.Framework.html) such as Open CL and afterwards selecting one or 
//! many available hardware to create a backend. A backend provides you with the functionality of 
//! managing the memory of the devices and copying memory objects to/from the host. Additionally, 
//! backends allow you to execute operations in parallel through kernel functions on the device(s) 
//! of the backend.
//!
//! # Architecture
//!
//! Backends are initialized by providing a framework and a selection of devices compatible with 
//! the framework to the [`Backend::new`](#method.new) associated function, or by simply 
//! calling [`Backend::default`](#method.default). The framework determines which devices are 
//! available and how parallel kernel functions can be executed.
//!
//! # Example
//!
//! ```
//! extern crate parenchyma;
//!
//! use parenchyma::frameworks::Native;
//! use parenchyma::prelude::*;
//!
//! // The `new` function initializes the framework on which it's called.
//! let framework: Native = Native::new().unwrap();
//! // The available frameworks can be obtained through the chosen `framework`.
//! let hardware = framework.hardware().to_vec();
//! // A ready to go backend can be created from the framework and hardware. It's worth noting that
//! // configuration options will be available in future versions.
//! let backend: Backend = Backend::with(framework, hardware).unwrap();
//! ```

use std::fmt;
use std::ops::Deref;

use super::compute_device::ComputeDevice;
use super::context::{Context, ContextCtor};
use super::error::{Error, ErrorKind, Result};
use super::extension_package::ExtensionPackage;
use super::framework::{Framework, FrameworkCtor};
use super::hardware::Hardware;

/// The representation of the backend.
pub struct Backend<Package = ()> {
    /// Provides the Framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be 
    /// used and determines which hardwares will be available and how parallel kernel 
    /// functions can be executed.
    framework: Box<Framework>,
    /// The context associated with the `framework`.
    ///
    /// Contexts are the heart of both OpenCL and CUDA applications. Contexts are created from one 
    /// or more devices that are capable of executing methods and synchronizing memory. See 
    /// the `Context` trait for more information.
    context: Box<Context<Package=Package>>,
    /// All _activatable_ hardware provided to the context.
    ///
    /// A cache of the hardware selection which is used as a representation of each framework's 
    /// list of available devices when selecting a new active device.
    selection: Vec<Hardware>
}

impl<P> Backend<P> where P: ExtensionPackage {
    /// Constructs a backend of the provided type with its default configurations. 
    ///
    /// # Return value
    ///
    /// The return value is a backend if the process goes well; otherwise, it returns 
    /// a simple error.
    pub fn new<F>() -> Result<Self>
        where F: FrameworkCtor,
              F::Context: ContextCtor<P,F=F> {

        let framework = F::new()?;
        let hardware = framework.hardware().to_vec();
        Self::with(framework, hardware)
    }

    /// Constructs a backend from the specified `framework` and `selection`.
    ///
    /// # Arguments
    ///
    /// * `framework` - One of the available frameworks.
    /// * `selection` - A selection of hardware provided by the specified `framework`.
    ///
    /// # Return value
    ///
    /// The return value is a backend if the process goes well; otherwise, it returns 
    /// a simple error.
    pub fn with<F>(framework: F, selection: Vec<Hardware>) -> Result<Self> 
        where F: FrameworkCtor, 
              F::Context: ContextCtor<P,F=F>, {

        info!("[PARENCHYMA] Constructing a backend using the {} framework", framework.name());
        let context = box F::Context::new(&framework, &selection)? as Box<Context<Package=P>>;
        let framework = box framework as Box<Framework>;
        Ok(Self { framework, context, selection })
    }
}

impl<P> Backend<P> where P: ExtensionPackage {
    /// Returns the active framework's active context's active device.
    pub fn active_device(&self) -> &dyn ComputeDevice {
        self.context.active_codev()
    }

    /// Simply returns the selected hardware.
    pub fn selection(&self) -> &[Hardware] {
        &self.selection
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
    pub fn select(&mut self, pred: &Fn(&Hardware) -> bool) -> Result {

        let nth = {
            self.selection().iter().enumerate()
                .filter(|&(_, h)| pred(h)).map(|(i, _)| i).nth(0)
        };

        match nth {
            Some(n) => self.context.activate(n),
            _ => {
                let message = "There are no devices matching the specified criteria.";
                Err(Error::new(ErrorKind::Other, message))
            }
        }
    }
    
    /// Synchronizes backend.
    pub fn synchronize(&self) -> Result {
        Ok(())
    }
}

impl<P> Deref for Backend<P> where P: ExtensionPackage {
    type Target = P::Extension;
    
    fn deref<'a>(&'a self) -> &'a Self::Target {
        self.context.extension()
    }
}

impl<E> fmt::Debug for Backend<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "A backend provided by the {} framework", self.framework.name())
    }
}
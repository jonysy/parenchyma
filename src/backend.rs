use super::{Context, Framework};
use super::error::{Error, Result};

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
///
/// ## Examples
///
/// ```rust
/// use parenchyma::{Backend, Framework, Native};
///
/// 
/// // Construct a new framework.
/// let framework = Native::new().expect("failed to initialize framework");
///
/// // Available devices can be obtained through the framework.
/// let selection = framework.available_devices[0].clone();
///
/// // Create a ready to go backend from the framework.
/// let backend = Backend::new(framework, selection).expect("failed to construct backend");
///
/// // ..
/// ```
///
/// Construct a default backend:
///
/// ```rust
/// use parenchyma::{Backend, Native};
///
/// // A default native backend.
/// let backend: Backend<Native> = Backend::default().expect("something went wrong!");
///
/// // ..
/// ```
#[derive(Debug)]
pub struct Backend<F: Framework> {
    framework: F,
    context: F::Context,
}

impl<F> Backend<F> where F: Framework {

    /// Construct a `Backend` from a [`framework`](./trait.Framework.html), such as OpenCL, CUDA, etc.,
    /// and a `selection` of devices.
    pub fn new(framework: F, selection: F::D) -> Result<Self> {

        let context = F::Context::new(selection).map_err(Error::from_framework::<F>)?;
        let backend = Backend { framework: framework, context: context};

        Ok(backend)
    }

    /// Constructs the default `Backend`.
    pub fn default() -> Result<Self> {
        let framework = F::new().map_err(Error::from_framework::<F>)?;
        let default_selection = framework.default_selection().remove(0);
        Self::new(framework, default_selection)
    }

    /// Returns the context.
    pub fn context(&self) -> &F::Context {
        &self.context
    }

    /// Returns the framework.
    pub fn framework(&self) -> &F {
        &self.framework
    }
}
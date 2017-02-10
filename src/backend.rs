use super::{Context, Framework};
use super::error::{Error, Result};

/// `Backend` is the heart of Parenchyma. `Backend` provides the interface for running parallel 
/// computations on one ore many devices.
///
/// This is the abstraction over which you are interacting with your devices. You can create a
/// `Backend` for computation by first choosing a specific `Framework` such as `OpenCL` and
/// afterwards selecting one or many available hardwares to create a `Backend`.
///
/// A `Backend` provides you with the functionality of managing the memory of the devices and copying
/// your objects from host to devices and the other way around. Additionally you can execute 
/// operations in parallel through kernel functions on the device(s) of the `Backend`.
///
/// ## Architecture
///
/// TODO..
#[derive(Debug)]
pub struct Backend<F> where F: Framework {
    /// The Framework implementation such as OpenCL, CUDA, etc., which should be used and
    /// determines which devices will be available and how parallel kernel functions can be
    /// executed.
    framework: F,
    /// A context, created from one or many devices, which are ready to execute kernel
    /// methods and synchronize memory.
    context: F::Context,
}

impl<F> Backend<F> where F: Framework {

    /// # Example
    ///
    /// ```rust
    /// use parenchyma::{Backend, Framework, Native};
    ///
    /// 
    /// // Construct a new framework.
    /// let framework = Native::new().expect("failed to initialize framework");
    ///
    /// // Available devices can be obtained through the framework.
    /// let selection = framework.available_devices.clone();
    ///
    /// // Create a ready to go `Backend` from the framework.
    /// let backend = Backend::new(framework, selection).expect("failed to construct backend");
    /// ```
    pub fn new(framework: F, selection: Vec<F::D>) -> Result<Self> {

        let context = F::Context::new(selection).map_err(Error::from_framework::<F>)?;
        let backend = Backend { framework: framework, context: context};

        Ok(backend)
    }

    /// # Example
    ///
    /// ```rust
    /// use parenchyma::{Backend, Native};
    ///
    /// let backend: Backend<Native> = Backend::default().expect("something went wrong!");
    /// ```
    pub fn default() -> Result<Self> {
        let framework = F::new().map_err(Error::from_framework::<F>)?;
        let default_selection = framework.default_selection();
        Self::new(framework, default_selection)
    }

    /// Returns the backend context.
    pub fn context(&self) -> &F::Context {
        &self.context
    }

    /// Returns the backend framework.
    pub fn framework(&self) -> &F {
        &self.framework
    }
}
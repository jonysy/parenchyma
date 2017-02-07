use super::{Context, Framework};
use super::error::Result;

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
    /// A context, created from one or many devices, which are ready to execute kernel
    /// methods and synchronize memory.
    context: F::Context,
    /// The Framework implementation such as OpenCL, CUDA, etc., which should be used and
    /// determines which devices will be available and how parallel kernel functions can be
    /// executed.
    framework: F,
}

impl<F> Backend<F> where F: Framework {//, Backend<F>: BackendExtn<F> {

    /// # Example
    ///
    /// ```rust,ignore
    /// extern crate parenchyma;
    /// extern crate parenchyma_native;
    ///
    /// use parenchyma::{Backend, Framework};
    /// use parenchyma_native::Native;
    ///
    /// 
    /// // Construct a new framework.
    /// let framework = Native::new();
    ///
    /// // Available devices can be obtained through the framework.
    /// let selection = framework.devices().to_vec();
    ///
    /// // Create a ready to go `Backend` from the framework.
    /// let backend = Backend::new(framework, selection).expect("Something went wrong!");
    /// ```
    pub fn new(framework: F, selection: Vec<F::Device>) -> Result<Self> {

        let context = F::Context::new(selection)?;
        let backend = Backend { framework: framework, context: context};

        Ok(backend)
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

pub trait BackendExtn<F: Framework> {

    /// # Example
    ///
    /// ```rust,ignore
    /// use parenchyma::Backend;
    /// use parenchyma_native::Native;
    ///
    /// let backend = Backend::<Native>::default().expect("Something went wrong!");
    /// ```
    fn default() -> Result<Backend<F>>;

    /// Synchronize backend.
    fn synchronize(&self) -> Result {

        Ok(())
    }
}
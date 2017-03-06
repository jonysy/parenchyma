use super::{Context, ComputeDevice, Framework};
use super::error::Result;

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
/// let selection = framework.available_devices.clone();
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
pub struct Backend {
    /// The initialized framework.
    pub framework: Box<Framework>, /* &'static str,*/
    /// The context associated with the `framework`.
    ///
    /// Contexts are the heart of both OpenCL and CUDA applications. See the [`Context`] trait for
    /// more information.
    ///
    /// [`Context`]: (./trait.Context.html)
    pub context: Box<Context>,
    /// The chosen device
    ///
    /// The default active device is the first device found (index = `0`).
    active: usize,
}

impl Backend {

    /// Constructs a backend using the most potent framework given the underlying hardware.
    pub fn new() -> Backend {

        unimplemented!()
    }

    /// Attempts to construct a backend from the specified `framework`.
    pub fn with<F>(framework: F) -> Result<Backend> where F: Framework {

        unimplemented!()
    }

    // /// Try all provided `frameworks` in the specified order, choosing the first framework that 
    // // initializes without failure.
    // pub fn try(frameworks: Vec<Box<Framework>>) -> Result<Backend>;
}

impl Backend {

    /// Returns the current device.
    pub fn compute_device<T>(&self) -> &ComputeDevice<T> {

        unimplemented!()
    }
}
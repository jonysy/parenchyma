use super::{Device, Result};
use super::opencl;

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
    context: opencl::OpenClContext,
    selected_device: usize,
}

impl Backend {

    pub fn new() -> Backend {
        // let framework = opencl::OpenCl::try_new()?;
        // let context = opencl::OpenClContext::try_from(framework.available_devices)?;

        // Ok(Backend { context, selected_device: 0 })

        unimplemented!()
    }

    pub fn try_from<F>(framework: F) -> Result<Backend> {

        unimplemented!()
    }

    pub fn context(&self) -> &opencl::OpenClContext {
        &self.context
    }

    pub fn device(&self) -> &opencl::OpenClDevice {

        &self.context.selection()[self.selected_device]
    }

    pub fn selected_device(&self) -> usize {
        self.selected_device
    }
}
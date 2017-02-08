use std::error::Error;
use super::Context;

/// Base functionality for all frameworks.
///
/// The default Framework would be a plain host CPU for common computation. To make use of other
/// computation hardware such as GPUs you would choose other computation Frameworks such as OpenCL
/// or CUDA, which provide the access to those hardwares for computation.
///
/// To start backend-agnostic and highly parallel computation, you start by initializing a
/// `Framework` implementation, resulting in an initialized framework that contains a list of
/// all available devices through that framework, as well as other things specific to the framework.
pub trait Framework: Sized {
    /// The name of the framework.
    ///
    /// Convention: <sup>*</sup> Use uppercase letters (e.g., `"OPEN_CL"`).
    const FRAMEWORK_NAME: &'static str;

    /// The context representation.
    type Context: Context<F = Self>;

    /// The device representation.
    type D;

    /// An error type associated with the framework.
    type E: 'static + Error + Send + Sync;

    /// The memory representation.
    ///
    /// Memory is allocated by a device in a way that it is accessible for its computations.
    type M;

    /// Initializes a new framework.
    fn new() -> Result<Self, Self::E>;

    /// Returns a default selection of devices for the framework.
    fn default_selection(&self) -> Vec<Self::D>;
}
use super::error::FrameworkSpecificError as Error;
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
    type Context: Context<D = Self::D>;

    /// The device representation.
    type D;

    /// An error type associated with the framework.
    type E: Error<F = Self>;

    /// Initializes a new framework.
    fn new() -> Result<Self, Self::E>;

    /// Returns a default selection of devices for the framework.
    fn default_selection(&self) -> Result<Vec<Self::D>, Self::E>;

    #[doc(hidden)]
    fn name() -> &'static str { /* /rust-lang/rust#29924 */ Self::FRAMEWORK_NAME }
}
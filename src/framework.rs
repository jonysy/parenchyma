use std::error::Error;
use super::Context;

/// A trait implemented for all frameworks. `Framework`s contain a list of all available devices as 
/// well as other objects specific to the implementor.
///
/// The default framework is simply the host CPU for common computation. To make use of other
/// devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to access the 
/// processing capabilities of the device(s).
pub trait Framework: Sized {
    /// The name of the framework.
    ///
    /// Naming convention: screaming snake case (e.g., `"OPEN_CL"`).
    const FRAMEWORK_NAME: &'static str;

    /// The context representation.
    type Context: Context<F = Self>;

    /// The device representation.
    type D;

    /// An error type associated with the framework.
    type E: 'static + Error + Send + Sync;

    /// The memory representation.
    type M;

    /// Initializes a new framework.
    fn new() -> Result<Self, Self::E>;

    /// Returns a default selection of devices available to the framework.
    fn default_selection(&self) -> Vec<Self::D>;
}
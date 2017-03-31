use context::{Context, ContextConfig};
use hardware::Hardware;
use std::fmt::Debug;

/// A trait implemented for all frameworks. `Framework`s contain a list of all available devices as 
/// well as other objects specific to the implementor.
///
/// The default framework is simply the host CPU for common computation. To make use of other
/// devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to access the 
/// processing capabilities of the device(s).
pub trait Framework: Debug {
    /// The name of the framework.
    ///
    /// This associated constant is mainly used for the purposes of debugging and reporting errors.
    ///
    /// **related issues**:
    ///
    /// [issue#29924](https://github.com/rust-lang/rust/issues/29924)
    const ID: &'static str;

    /// Returns the cached and available hardware.
    fn hardware(&self) -> &[Hardware];
}

/// Extends the framework with useful associated types and functions.
///
/// **note**:
///
/// This trait isn't object-safe, therefore can only be used during backend initialization, 
/// memory access, etc..
pub trait FrameworkExt: Sized {

    /// TODO use `TryFrom` when stable.
    ///
    /// [#33417](https://github.com/rust-lang/rust/issues/33417)
    type Context: Context + for<'a> From<ContextConfig<'a, Self>>;
}
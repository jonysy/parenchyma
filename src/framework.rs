use hardware::Hardware;
use std::fmt::Debug;
use utility::typedef::Result;

/// A trait implemented for all frameworks. `Framework`s contain a list of all available 
/// devices as well as other objects specific to the implementor.
///
/// The default framework is simply the host CPU for common computation. To make use 
/// of other devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to 
/// access the processing capabilities of the device(s).
pub trait Framework: Debug {
    /// The name of the framework.
    ///
    /// This associated constant is mainly used for the purposes of debugging and reporting errors.
    ///
    /// **related issues**:
    ///
    /// [issue#29924](https://github.com/rust-lang/rust/issues/29924)
    const ID: &'static str;

    /// Initializes the framework.
    fn new() -> Result<Self> 
        where Self: Sized {

        unimplemented!()
    }

    /// Returns the cached and available hardware.
    fn hardware(&self) -> &[Hardware];
}
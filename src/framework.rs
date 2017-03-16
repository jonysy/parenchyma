use std::fmt::Debug;

use super::{Context, Hardware, Result};

/// A trait implemented for all frameworks. `Framework`s contain a list of all available devices as 
/// well as other objects specific to the implementor.
///
/// The default framework is simply the host CPU for common computation. To make use of other
/// devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to access the 
/// processing capabilities of the device(s).
pub trait Framework: 'static + Debug {
    /// The name of the framework.
    ///
    /// This associated constant is mainly used for the purposes of debugging and reporting errors.
    ///
    /// [issue#29924](https://github.com/rust-lang/rust/issues/29924): remove `Framework::name`
    const FRAMEWORK_NAME: &'static str;

    /// Returns the cached and available hardware.
    fn available_hardware(&self) -> Vec<Hardware>;
}

/// Initialize a context, box it, and then return it.
pub trait BoxContext<ExtensionPackage>: Framework {

    // /// The `Context` representation for this framework.
    // type Context: TryFrom<ContextConfig<Self>, Err = Error>;

    /// Create a context from a selection of hardware devices and then wrap it in a box.
    fn enclose(&self, selection: Vec<Hardware>) 
        -> Result<Box<Context<Package = ExtensionPackage>>>;
}
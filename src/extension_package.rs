pub use self::backward::Backward;
pub use self::configuration::{ConvolutionConfiguration, LrnConfiguration, PoolingConfiguration};
pub use self::convolution::{ConvBackwardDataAlgo, ConvBackwardFilterAlgo, ConvForwardAlgo};
pub use self::forward::Forward;

mod backward;
mod configuration;
mod convolution;
mod forward;

use parenchyma::extension_package::ExtensionPackage;

/// The BLAS package.
pub enum Package {
    // ..
}

/// Provides the functionality for a backend to support DNN related operations.
pub trait Extension: Backward + Forward {
    // ..
}

impl ExtensionPackage for Package {
    type Extension = Extension;

    fn package_name(&self) -> &'static str {
        return "parenchyma/deep";
    }
}
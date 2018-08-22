use parenchyma::extension_package::{Dependency, ExtensionPackage};
use super::{parenchyma_blas, parenchyma_deep};

/// The machine learning package.
pub struct Package {
    /// The BLAS package.
    pub(crate) blas: parenchyma_blas::Package,
    /// The Deep NN package.
    pub(crate) deep: parenchyma_deep::Package,
}

impl Dependency<parenchyma_blas::Package> for Package {
    fn dependency(&self) -> &parenchyma_blas::Package {
        &self.blas
    }
}

impl Dependency<parenchyma_deep::Package> for Package {
    fn dependency(&self) -> &parenchyma_deep::Package {
        &self.deep
    }
}

/// **note**: should be replaced with an actual trait alias ([RFC#1733]).
///
/// [RFC#1733]: https://github.com/rust-lang/rfcs/pull/1733
pub trait Dependencies: 
    Dependency<parenchyma_blas::Package> + 
    Dependency<parenchyma_deep::Package> {
    //..
}

impl<D> Dependencies for D
    where D:
    Dependency<parenchyma_blas::Package> + 
    Dependency<parenchyma_deep::Package> {
    // ..
}

pub trait Extension 
    where Self: 
    parenchyma_blas::Extension + 
    parenchyma_deep::Extension {
    // ..
}

impl ExtensionPackage for Package {
    type Extension = Extension;
    fn package_name(&self) -> &'static str {
        return "parenchyma/ml";
    }
}
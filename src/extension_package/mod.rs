pub use self::axpby::Axpby;
pub use self::level1::Vector;
pub use self::level2::MatrixVector;
pub use self::level3::Matrix;
pub use self::transpose::Transposition;

mod axpby;
mod level1;
mod level2;
mod level3;
mod transpose;

use parenchyma::extension_package::ExtensionPackage;

/// The BLAS package.
pub enum Package {
    Native,
    OpenCL(::frameworks::open_cl::OpenCLPackage),
}

impl Package {
    pub fn open_cl(&self) -> &::frameworks::open_cl::OpenCLPackage {
        if let &Package::OpenCL(ref package) = self {
            package
        } else {
            panic!("an Open CL package was expected, but another package was found.")
        }
    }
}

/// Provides level 1, 2, and 3 BLAS operations.
///
/// **note**: should be replaced with an actual trait alias ([RFC#1733]).
///
/// [RFC#1733]: https://github.com/rust-lang/rfcs/pull/1733
pub trait Extension: Axpby + Vector + MatrixVector + Matrix { }

impl ExtensionPackage for Package {
    type Extension = Extension;

    fn package_name(&self) -> &'static str {
        return "parenchyma/blas";
    }
}
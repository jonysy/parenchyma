use super::super::{Dependencies, Extension, Package};

use parenchyma::error::Result;
use parenchyma::extension_package::ExtensionPackageCtor;
use parenchyma::frameworks::OpenCLContext as Context;
use parenchyma_blas::Package as BLASPackage;
use parenchyma_blas::frameworks::open_cl::OpenCLPackage as OpenCLBLASPackage;
use parenchyma_deep::Package as DeepPackage;
use parenchyma_deep::frameworks::open_cl::OpenCLPackage as OpenCLDeepPackage;

impl<P> Extension for Context<P> where P: Dependencies { }

impl ExtensionPackageCtor<Context<()>> for Package {
    fn package(target: &mut Context<()>) -> Result<Self> {
        let blas = OpenCLBLASPackage::compile(target).map(BLASPackage::OpenCL)?;
        let deep = OpenCLDeepPackage::compile(target).map(DeepPackage::OpenCL)?;

        Ok(Package { blas, deep })
    }
}
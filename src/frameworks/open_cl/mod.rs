//! TODO
//!
//! If possible, use 'fast-versions'. note: 'fast-versions' have specific requirements.
//!
//! [Cloning kernels before arguments are specified. #97][1]
//!
//! [1](https://github.com/cogciprocate/ocl/issues/97#issuecomment-367178247)

pub use self::package::OpenCLPackage;

mod package;

use parenchyma::error::{Error, ErrorKind, Result};
use parenchyma::extension_package::{Dependency, ExtensionPackageCtor};
use parenchyma::frameworks::{OpenCLContext as Context, OpenCLMemory as Memory};
use parenchyma::tensor::{self, SharedTensor};
use ocl;

use super::super::{Extension, Package, Transposition};
use super::super::extension_package::{Matrix, MatrixVector, Vector};

impl<P> Extension for Context<P> where P: Dependency<Package> {
    // ..
}

impl<P> Vector for Context<P> where P: Dependency<Package> {
    fn asum(&self, x: &SharedTensor<f32>, result: &mut SharedTensor<f32>) -> Result {
        let length = x.shape().capacity();
        let offset = 0i32;
        let inc = 1i32;

        unimplemented!()
    }

    fn axpy(&self, a: &SharedTensor, x: &SharedTensor, y: &mut SharedTensor) -> Result {

        let n = x.shape().capacity;
        let offset = 0;
        let inc = 1;

        let alpha: &Memory<_> = tensor::reference(a, /*on:*/ self.device())?;
        let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
        let y: &mut Memory<_> = tensor::mut_reference(y, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("Xaxpy", &self.extension_package().dependency().open_cl().program)?
                .arg_scl(n)
                .arg_buf(alpha)
                .arg_buf(x).arg_scl(offset).arg_scl(inc)
                .arg_buf(y).arg_scl(offset).arg_scl(inc)
                // .gwo(..)
                .gws([64, 1, 1])
                .lws([64, 1, 1])
                // todo The queue must be associated with a device associated with 
                // the kernel's program.
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }
}

impl<P> Matrix for Context<P> where P: Dependency<Package> {
    // ..
}

impl<P> MatrixVector for Context<P> where P: Dependency<Package> {
    // ..
}

impl ExtensionPackageCtor<Context> for super::super::Package {
    fn package(target: &mut Context) -> Result<Self> {
        OpenCLPackage::compile(target).map(Package::OpenCL)
    }
}
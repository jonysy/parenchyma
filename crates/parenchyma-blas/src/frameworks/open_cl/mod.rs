//! TODO
//!
//! If possible, use 'fast-versions'. note: 'fast-versions' have specific requirements.
//!
//! [Cloning kernels before arguments are specified. #97][1]
//!
//! limits: https://stackoverflow.com/a/40272984/8034246
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
            // take into account the CL_DEVICE_MAX_WORK_GROUP_SIZE (given by clGetDeviceInfo()) 
            // parameter. For bigger matrices, you will have to use more than 2 dimensions.
            //
            // The global size (GSZ) is the total number of work-items (WI)
            // The local size (LSZ) is the number of work-items per work-group (WI/WG)
            // The number of work-groups is the global size / local size, or GSZ/LSZ, or WG

            ocl::Kernel::new("Xaxpy", &self.extension_package().dependency().open_cl().program)?
                .arg_scl(n)
                .arg_buf(alpha)
                .arg_buf(x).arg_scl(offset).arg_scl(inc)
                .arg_buf(y).arg_scl(offset).arg_scl(inc)
                // .gwo(..)
                // .gws([64 * 4,   1, 1])
                .gws([64,   1, 1])
                .lws([64,       1, 1])

                // todo The queue must be associated with a device associated with 
                // the kernel's program.
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }

    fn copy(&self, from: &SharedTensor, to: &mut SharedTensor) -> Result {
        let length = from.shape().capacity();
        let offset = 0;
        let inc = 1;

        let from: &Memory<_> = tensor::reference(from, /*on:*/ self.device())?;
        let to: &mut Memory<_> = tensor::mut_reference(to, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("Xcopy", &self.extension_package().dependency().open_cl().program)?
                .arg_scl(length as i32)
                .arg_buf(from)
                .arg_scl(offset)
                .arg_scl(inc)
                .arg_buf(to)
                .arg_scl(offset)
                .arg_scl(inc)

                .gws([64, 1, 1])
                .lws([64, 1, 1])

                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }

    fn scal(&self, a: &SharedTensor, x: &mut SharedTensor) -> Result {
        let length = x.shape().capacity();
        let offset = 0;
        let inc = 1;

        let a: &Memory<_> = tensor::reference(a, /*on:*/ self.device())?;
        let x: &mut Memory<_> = tensor::mut_reference(x, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("Xscal", &self.extension_package().dependency().open_cl().program)?
                .arg_scl(length as i32)
                .arg_buf(a)
                .arg_buf(x)
                .arg_scl(offset)
                .arg_scl(inc)

                .gws([64, 1, 1])
                .lws([64, 1, 1])

                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }
}

impl<P> Matrix for Context<P> where P: Dependency<Package> {
    /// Provides the gemm operation.
    ///
    /// Computes a matrix-matrix product with general matrices.
    fn gemm(
        self: &Self,
        alpha: &SharedTensor,
        amatrix_transposition: Transposition,
        amatrix: &SharedTensor,
        bmatrix_transposition: Transposition,
        bmatrix: &SharedTensor,
        beta: &SharedTensor,
        cmatrix: &mut SharedTensor) -> Result {

        const WGD: usize = 8; // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
        const MDIMCD: usize = 8; // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
        const NDIMCD: usize = 8; // Threads per workgroup in N-dimension (e.g. 8, 16, 32)

        // TODO
        // 1) check that `c` has the correct `shape`

        // rounding functions performing ceiling and division operations
        fn ceil_div(x: usize, y: usize) -> usize { 1 + ((x - 1) / y) }
        fn ceil(x: usize, y: usize) -> usize { ceil_div(x, y) * y }

        let column_major = false;
        let row_major = true;
        let offset = 0;

        let a_nrows = amatrix.shape().dimensions()[0];
        //let a_ncols = amatrix.shape().dimensions()[1];
        let a_ncols = amatrix.shape().dimensions().iter().skip(1).fold(1, |prod, d| prod * d); // ..?

        let b_nrows = bmatrix.shape().dimensions()[0];
        //let b_ncols = bmatrix.shape().dimensions()[1];
        let b_ncols = bmatrix.shape().dimensions().iter().skip(1).fold(1, |prod, d| prod * d); // ..?

        let c_nrows = cmatrix.shape().dimensions()[0];
        //let c_ncols = cmatrix.shape().dimensions()[1];
        let c_ncols = cmatrix.shape().dimensions().iter().skip(1).fold(1, |prod, d| prod * d); // ..?

        let n = match bmatrix_transposition {
            Transposition::NoTranspose => b_ncols,
            _ => b_nrows
        };

        let (m, k) = match amatrix_transposition {
            Transposition::NoTranspose => (a_nrows, a_ncols),
            _ => (a_ncols, a_nrows)
        };

        // row-major: distance between two consecutive rows
        // col-major: distance between two consecutive columns
        let a_leading = a_ncols;
        let b_leading = b_ncols;
        let c_leading = c_ncols;

        // =============================

        // **important**:
        //
        // > Computes whether or not the matrices are transposed in memory. This is based on 
        // > their layout (row or column-major) and whether or not they are requested to 
        // > be pre-transposed. Note that the Xgemm kernel expects either matrices A 
        // > and C (in case of row-major) or B (in case of col-major) to be transformed, so 
        // > transposing requirements are not the same as whether or not the matrix is actually 
        // > transposed in memory.

        let a_rotated = 
            (column_major && amatrix_transposition != Transposition::NoTranspose) || 
            (row_major && amatrix_transposition == Transposition::NoTranspose);

        let b_rotated = 
            (column_major && bmatrix_transposition != Transposition::NoTranspose) || 
            (row_major && bmatrix_transposition == Transposition::NoTranspose);

        let c_rotated = row_major == true;
        let a_want_rotated = false;
        let b_want_rotated = true;
        let c_want_rotated = false;
        let a_do_transpose = a_rotated != a_want_rotated;
        let b_do_transpose = b_rotated != b_want_rotated;
        let c_do_transpose = c_rotated != c_want_rotated;

        // In case of complex data-types, the transpose can also become a conjugate transpose
        let a_conjugate = amatrix_transposition == Transposition::ConjugateTranspose;
        let b_conjugate = bmatrix_transposition == Transposition::ConjugateTranspose;

        unsafe {
            // Retrieves the proper XgemmDirect kernel from the compiled binary
            let kernel = {
                if a_do_transpose {
                    if b_do_transpose {
                        ocl::Kernel::new(
                            "XgemmDirectTT", 
                            &self.extension_package().dependency().open_cl().program
                        )?
                    } else {
                        ocl::Kernel::new(
                            "XgemmDirectTN", 
                            &self.extension_package().dependency().open_cl().program
                        )?
                    }
                } else {
                    if b_do_transpose {
                        ocl::Kernel::new(
                            "XgemmDirectNT", 
                            &self.extension_package().dependency().open_cl().program
                        )?
                    } else {
                        ocl::Kernel::new(
                            "XgemmDirectNN", 
                            &self.extension_package().dependency().open_cl().program
                        )?
                    }
                }
            };

            // compute the global and local thread sizes
            let m_ceiled = ceil(m, WGD);
            let n_ceiled = ceil(n, WGD);
            let global = &[(m_ceiled * MDIMCD) / WGD, (n_ceiled * NDIMCD) / WGD];
            // let global = [
            //     ((1 + ((m - 1) / WGD)) * WGD * m_ceiled) / WGD,
            //     ((1 + ((n - 1) / WGD)) * WGD * n_ceiled) / WGD,
            //     1
            // ];
            let local = &[MDIMCD, NDIMCD];

            // set the kernel arguments
            kernel
                .arg_scl(m as i32)
                .arg_scl(n as i32)
                .arg_scl(k as i32)
                .arg_buf(tensor::reference(alpha, /*on:*/ self.device())?: &Memory<_>)
                .arg_buf(tensor::reference(beta, /*on:*/ self.device())?: &Memory<_>)
                .arg_buf(tensor::reference(amatrix, /*on:*/ self.device())?: &Memory<_>)
                .arg_scl(offset as i32)
                .arg_scl(a_leading as i32)
                .arg_buf(tensor::reference(bmatrix, /*on:*/ self.device())?: &Memory<_>)
                .arg_scl(offset as i32)
                .arg_scl(b_leading as i32)
                .arg_buf(tensor::mut_reference(cmatrix, /*on:*/ self.device())?: &mut Memory<_>)
                .arg_scl(offset as i32)
                .arg_scl(c_leading as i32)
                .arg_scl(c_do_transpose as i32)
                .arg_scl(a_conjugate as i32)
                .arg_scl(b_conjugate as i32)

                .gws(global)
                .lws(local)
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }
}

impl<P> MatrixVector for Context<P> where P: Dependency<Package> {
    // ..
}

impl ExtensionPackageCtor<Context<()>> for super::super::Package {
    fn package(target: &mut Context<()>) -> Result<Self> {
        OpenCLPackage::compile(target).map(Package::OpenCL)
    }
}
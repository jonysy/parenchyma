use parenchyma::error::{Error, ErrorKind, Result};
use parenchyma::extension_package::Dependency;
use parenchyma::frameworks::NativeContext as Context;
use parenchyma::tensor::SharedTensor;

use rblas;
use rblas::math::mat::Mat;
use rblas::matrix::Matrix as IMatrix;

use super::super::{Extension, Package, Transposition};
use super::super::extension_package::{Matrix, MatrixVector, Vector};

impl<P> Extension for Context<P> where P: Dependency<Package> { }

impl<P> Vector for Context<P> where P: Dependency<Package> {
    fn asum(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        result.as_mut_slice_unsynched()?[0] = rblas::Asum::asum(x.as_slice()?);
        Ok(())
    }

    fn axpy(&self, a: &SharedTensor, x: &SharedTensor, y: &mut SharedTensor) -> Result {
        Ok(rblas::Axpy::axpy(
            a.as_slice()?.get(0)
                .ok_or_else(|| Error::new(ErrorKind::Other, "Index out of bounds"))?, 
            x.as_slice()?, 
            y.as_mut_slice()?
        ))
    }

    fn copy(&self, from: &SharedTensor, to: &mut SharedTensor) -> Result {
        Ok(rblas::Copy::copy(
            from.as_slice()?, to.as_mut_slice_unsynched()?))
    }

    fn dot(&self, x: &SharedTensor, y: &SharedTensor, result: &mut SharedTensor) -> Result {
        result.as_mut_slice_unsynched()?[0] =  rblas::Dot::dot(x.as_slice()?, y.as_slice()?);
        Ok(())
    }

    fn nrm2(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        result.as_mut_slice_unsynched()?[0] =  rblas::Nrm2::nrm2(x.as_slice()?);
        Ok(())
    }

    fn scal(&self, a: &SharedTensor, x: &mut SharedTensor) -> Result {
        Ok(rblas::Scal::scal(
            a.as_slice()?.get(0)
                .ok_or_else(|| Error::new(ErrorKind::Other, "Index out of bounds"))?, 
            x.as_mut_slice()?
        ))
    }

    fn swap(&self, x: &mut SharedTensor, y: &mut SharedTensor) -> Result {
        Ok(rblas::Swap::swap(x.as_mut_slice()?, y.as_mut_slice()?))
    }
}

impl<P> Matrix for Context<P> where P: Dependency<Package> {
    fn gemm(
        self: &Self,
        alpha: &SharedTensor,
        amatrix_transposition: Transposition,
        amatrix: &SharedTensor,
        bmatrix_transposition: Transposition,
        bmatrix: &SharedTensor,
        beta: &SharedTensor,
        cmatrix: &mut SharedTensor) -> Result {

        let a_0 = amatrix.shape().dimensions()[0] as i32;
        let a_1 = amatrix.shape().dimensions().iter().skip(1).fold(1, |prod, i| prod * i) as i32;

        let b_0 = bmatrix.shape().dimensions()[0] as i32;
        let b_1 = bmatrix.shape().dimensions().iter().skip(1).fold(1, |prod, i| prod * i) as i32;

        let c_0 = cmatrix.shape().dimensions()[0] as i32;
        let c_1 = cmatrix.shape().dimensions().iter().skip(1).fold(1, |prod, i| prod * i) as i32;

        let input = as_matrix(amatrix.as_slice()?, a_0 as usize, a_1 as usize);
        let weights = as_matrix(bmatrix.as_slice()?, b_0 as usize, b_1 as usize);
        let mut output = as_matrix(cmatrix.as_slice()?, c_0 as usize, c_1 as usize);

        rblas::Gemm::gemm(
            &alpha.as_slice()?[0], 
            amatrix_transposition.into(), 
            &input, 

            bmatrix_transposition.into(),
            &weights,
            &beta.as_slice()?[0], 

            &mut output
        );

        read_from_matrix(&output, cmatrix.as_mut_slice()?);

        Ok(())
    }
}

fn as_matrix(slice: &[f32], nrows: usize, ncols: usize) -> Mat<f32> {
    let mut mat: Mat<f32> = Mat::new(nrows, ncols);

    for i in 0..nrows {
        for j in 0..ncols {
            let index = ncols * i + j;
            unsafe {
                *mat.as_mut_ptr().offset(index as isize) = slice[index].clone();
            }
        }
    }

    mat
}

fn read_from_matrix(mat: &Mat<f32>, slice: &mut [f32]) {
    let n = mat.rows();
    let m = mat.cols();
    for i in 0..n {
        for j in 0..m {
            let index = m * i + j;
            slice[index] = mat[i][j].clone();
        }
    }
}

impl<P> MatrixVector for Context<P> where P: Dependency<Package> { }
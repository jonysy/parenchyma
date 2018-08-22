use parenchyma::error::Result;
use parenchyma::tensor::SharedTensor;

use super::Transposition;

// pub struct View<'a>(&'a Array<f32, IxDyn>);
// pub struct ViewMut<'a>(&'a mut Array<f32, IxDyn>);
// impl<'a> Matrix<f32> for View<'a> {
//     fn rows(&self) -> i32 {
//         self.0.rows()
//     }
//     fn cols(&self) -> i32 {
//         self.0.cols()
//     }
//     fn as_ptr(&self) -> *const f32 {
//         unimplemented!()
//     }
//     fn as_mut_ptr(&self) -> *mut f32 {
//         unimplemented!()
//     }
// }

pub struct GenericMatrix<'a> {
    /// The factor of matrix A (scalar).
    pub scalar: &'a SharedTensor,
    /// Buffer object storing matrix A.
    pub matrix: &'a SharedTensor,
    /// How matrix A is to be transposed.
    pub transposition: Transposition,
}

/// The trait `Matrix` consists of level 3 BLAS routines - matrix-matrix operations, including a 
/// general matrix multiplication.
pub trait Matrix {
    /// Computes a matrix-matrix product with general matrices.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The factor of matrix A (scalar).
    /// * `amatrix_transposition` - How matrix A is to be transposed.
    /// * `amatrix` - The buffer object storing matrix A..
    fn gemm(
        self: &Self,
        alpha: &SharedTensor,
        amatrix_transposition: Transposition,
        amatrix: &SharedTensor,
        bmatrix_transposition: Transposition,
        bmatrix: &SharedTensor,
        beta: &SharedTensor,
        cmatrix: &mut SharedTensor) -> Result {
        unimplemented!()
    }
}
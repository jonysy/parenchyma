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

/// `Matrix` consists of level 3 BLAS routines - matrix-matrix operations, including a general 
/// matrix multiplication.
pub trait Matrix {
    /// Provides the gemm operation.
    ///
    /// Computes a matrix-matrix product with general matrices.
    fn gemm(
        &self, 
        _alpha: &SharedTensor, 
        _a_transpose: Transposition, 
        _a: &SharedTensor, 
        _b_transpose: Transposition, 
        _b: &SharedTensor, 
        _beta: &SharedTensor, 
        _c: &mut SharedTensor) -> Result {
        unimplemented!()
    }
}
use parenchyma::error::Result;
use parenchyma::tensor::SharedTensor;

/// `Vector` consists of level 1 BLAS routines - vector operations on strided arrays.
pub trait Vector {
    /// Provides the asum operation.
    ///
    /// Computes the sum of the absolute values of the elements of `x`, and the saves the `result`.
    fn asum(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!("asum")
    }
    /// Provides the axpy operation.
    ///
    /// Computes a vector `x` times a constant `a` plus a vector `y` (i.e., `a * x + y`), and then
    /// saves the result to `y`.
    fn axpy(&self, a: &SharedTensor, x: &SharedTensor, y: &mut SharedTensor) -> Result {
        unimplemented!("axpy")
    }
    /// Provides the copy operation.
    ///
    /// Copies `from.len()` elements of vector `from` into vector `to`.
    fn copy(&self, from: &SharedTensor, to: &mut SharedTensor) -> Result {
        unimplemented!("copy")
    }
    /// Provides the dot operation.
    ///
    /// Computes the [dot product] over `x` and `y`, and then saves the `result`.
    fn dot(&self, x: &SharedTensor, y: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!("dot")
    }
    /// Provides the nrm2 operation.
    ///
    /// Computes the L2 norm (i.e., the euclidean length of vector `x`), and then saves the `result`.
    fn nrm2(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!("nrm2")
    }
    /// Provides the scal operation.
    ///
    /// Scales a vector `x` by a constant `a` (i.e., `a * x`).
    fn scal(&self, a: &SharedTensor, x: &mut SharedTensor) -> Result {
        unimplemented!("scal")
    }
    /// Provides the swap operation.
    ///
    /// Swaps the elements of vector `x` and vector `y`.
    fn swap(&self, x: &mut SharedTensor, y: &mut SharedTensor) -> Result {
        unimplemented!("swap")
    }
}
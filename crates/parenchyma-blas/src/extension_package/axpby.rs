use parenchyma::error::Result;
use parenchyma::tensor::SharedTensor;

/// Extends IBlas with Axpby
pub trait Axpby: super::Vector {
    /// Performs the operation y := a*x + b*y .
    ///
    /// Consists of a scal(b, y) followed by a axpby(a,x,y).
    fn axpby(&self, a: &SharedTensor, x: &SharedTensor, b: &SharedTensor, y: &mut SharedTensor) -> Result {
        self.scal(b, y)?;
        self.axpy(a, x, y)?;
        Ok(())
    }
}

impl<A> Axpby for A where A: super::Vector {
    // ..
}
use parenchyma::error::Result;
use parenchyma::tensor::SharedTensor;

/// Extends IBlas with Axpby
pub trait Axpby: super::Vector {
    /// Performs the operation y := a*x + b*y .
    ///
    /// Consists of a scal(b, y) followed by a axpby(a,x,y).
    fn axpby(&self, a: &SharedTensor, x: &SharedTensor, b: &SharedTensor, y: &mut SharedTensor) -> Result {
        // try!(self.scal_plain(b, y));
        // try!(self.axpy_plain(a, x, y));
        // Ok(())

        unimplemented!()
    }
}

impl<A> Axpby for A where A: super::Vector {
    // ..
}
use parenchyma::error::{Error, ErrorKind, Result};
use parenchyma::extension_package::Dependency;
use parenchyma::frameworks::NativeContext as Context;
use parenchyma::tensor::SharedTensor;
use rblas;
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
        &self, 
        alpha: &SharedTensor, 
        a_transpose: Transposition, 
        a: &SharedTensor, 
        b_transpose: Transposition, 
        b: &SharedTensor, 
        beta: &SharedTensor, 
        c: &mut SharedTensor) -> Result {
        unimplemented!()
    }
}

impl<P> MatrixVector for Context<P> where P: Dependency<Package> { }
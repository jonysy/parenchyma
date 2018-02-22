use parenchyma;
use parenchyma::open_cl::OpenCLContext;
use parenchyma::{Context, SharedTensor};
use parenchyma::error::Result;

use extension::Vector;

impl Vector for OpenCLContext {

    fn asum(
        &self, 
        x: &SharedTensor<f32>, 
        result: &mut SharedTensor<f32>) -> Result {

        unimplemented!()
    }

    fn axpy(
        &self, 
        a: &SharedTensor<f32>, 
        x: &SharedTensor<f32>, 
        y: &mut SharedTensor<f32>) -> Result {

        let kernel: ::ocl::Kernel = unimplemented!();

        let n = x.shape().capacity;

        let alpha = parenchyma::tensor(self, alpha)?;
        let x = parenchyma::tensor(self, x)?;
        let y = parenchyma::tensor_mut(self, y)?;

        let offset = 0;
        let inc = 1;

        kernel
            .arg_scl(n)
            .arg_buf(alpha)
            .arg_buf(x).arg_scl(offset).arg_scl(inc)
            .arg_buf(y).arg_scl(offset).arg_scl(inc)
        //     //.gwo(..)
        //     .gws([WGS, 1, 1])
        //     .lws([WGS, 1, 1])
        //     // todo The queue must be associated with a device associated with the kernel's program.
            .queue(self.active_direct().queue().clone())
            .enq()?;


        Ok(())
    }

    fn copy(
        &self, 
        from: &SharedTensor<f32>, 
        to: &mut SharedTensor<f32>) -> Result {

        unimplemented!()
    }

    fn dot(
        &self, 
        x: &SharedTensor<f32>, 
        y: &SharedTensor<f32>, 
        result: &mut SharedTensor<f32>) -> Result {

        unimplemented!()
    }

    fn nrm2(
        &self, 
        x: &SharedTensor<f32>, 
        result: &mut SharedTensor<f32>) -> Result {

        unimplemented!()
    }

    fn scal(
        &self, 
        a: &SharedTensor<f32>, 
        x: &mut SharedTensor<f32>) -> Result {

        unimplemented!()
    }

    fn swap(
        &self, 
        x: &mut SharedTensor<f32>, 
        y: &mut SharedTensor<f32>) -> Result {

        unimplemented!()
    }
}
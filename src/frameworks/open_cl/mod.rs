pub use self::package::OpenCLPackage;

mod package;

use super::super::{Extension, Package};
use super::super::extension_package::{Backward, Forward};

use ocl;
use parenchyma::error::Result;
use parenchyma::extension_package::{Dependency, ExtensionPackageCtor};
use parenchyma::frameworks::{OpenCLContext as Context, OpenCLMemory as Memory};
use parenchyma::tensor::{self, SharedTensor};

impl ExtensionPackageCtor<Context<()>> for super::super::Package {
    fn package(target: &mut Context<()>) -> Result<Self> {
        OpenCLPackage::compile(target).map(Package::OpenCL)
    }
}

impl<P> Backward for Context<P> where 
    P: Dependency<Package> {
    fn log_softmax_grad(
        &self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result: &mut SharedTensor) -> Result {

        let n = x.shape().capacity;
        let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
        let x_diff: &Memory<_> = tensor::reference(x_diff, /*on:*/ self.device())?;
        let result: &mut Memory<_> = tensor::mut_reference(result, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("log_softmax_backward_float", &self.extension_package().dependency().open_cl().program)?
                .arg_buf(x)
                .arg_buf(x_diff)
                .arg_buf(result)
                .arg_scl(n as i32)

                .gws([1, 1, 1])
                .lws([1, 1, 1])
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }

    // fn relu_grad(
    //     self: &Self, 
    //     x: &SharedTensor, 
    //     x_diff: &SharedTensor,
    //     result: &SharedTensor,
    //     result_diff: &mut SharedTensor) -> Result {
    //     let res = x.as_slice().unwrap().iter()
    //         .zip(x_diff.as_slice().unwrap().iter())
    //         .map(|(x, dx)| if *x > 0.0 { *dx } else { 0.0 });
    //     result_diff.write_iter(res)?;
    //     Ok(())
    // }

    fn sigmoid_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor,
        _: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {

        let n = x.shape().capacity;
        let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
        let x_diff: &Memory<_> = tensor::reference(x_diff, /*on:*/ self.device())?;
        // let result: &Memory<_> = tensor::reference(result, /*on:*/ self.device())?;
        let result_diff: &mut Memory<_> = tensor::mut_reference(result_diff, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("sigmoid_backward_float", &self.extension_package().dependency().open_cl().program)?
                .arg_buf(x)
                .arg_buf(x_diff)
                .arg_buf(result_diff)
                .arg_scl(n as i32)

                .gws([n])
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }

    // fn softmax_grad(
    //     self: &Self,
    //     x: &SharedTensor, 
    //     x_diff: &SharedTensor, 
    //     result_diff: &mut SharedTensor) -> Result {
    //     let mut dot = 0.0;
    //     let sig_data_slice = x.as_slice().unwrap();
    //     let sig_dx_slice = x_diff.as_slice().unwrap();
    //     for (t, dt) in sig_data_slice.iter().zip(sig_dx_slice.iter()) {
    //         dot += t * dt;
    //     }
    //     let res = sig_data_slice.iter().zip(sig_dx_slice.iter()).map(|(t, dt)| t * (dt - dot));
    //     result_diff.write_iter(res)?;
    //     Ok(())
    // }

    // fn tanh_grad(
    //     self: &Self, 
    //     x: &SharedTensor, 
    //     x_diff: &SharedTensor, 
    //     result: &SharedTensor,
    //     result_diff: &mut SharedTensor) -> Result {
    //     let res = x.as_slice().unwrap().iter()
    //         .zip(x_diff.as_slice().unwrap().iter())
    //         .map(|(x, dx)| (1.0 - x.powi(2)) * *dx);
    //     result_diff.write_iter(res)?;
    //     Ok(())
    // }
}

impl<P> Forward for Context<P> where 
    P: Dependency<Package> {
    // fn elu(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
    //     let n = x.shape().capacity;
    //     let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
    //     let result: &mut Memory<_> = tensor::mut_reference(result, /*on:*/ self.device())?;

    //     unsafe {
    //         ocl::Kernel::new("elu_float", &self.extension_package().dependency().open_cl().program)?
    //             .arg_buf(x)
    //             .arg_buf(result)
    //             .arg_scl(n as i32)

    //             .gws([n])
    //             .queue(self.device().queue().clone())
    //             .enq()?;
    //     }

    //     Ok(())
    // }

    fn log_softmax(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let n = x.shape().capacity;
        let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
        let result: &mut Memory<_> = tensor::mut_reference(result, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("log_softmax_float", &self.extension_package().dependency().open_cl().program)?
                .arg_buf(x)
                .arg_buf(result)
                .arg_scl(n as i32)

                .gws([1, 1, 1])
                .lws([1, 1, 1])
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }

    // fn relu(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
    //     let n = x.shape().capacity;
    //     let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
    //     let result: &mut Memory<_> = tensor::mut_reference(result, /*on:*/ self.device())?;

    //     unsafe {
    //         ocl::Kernel::new("relu_float", &self.extension_package().dependency().open_cl().program)?
    //             .arg_buf(x)
    //             .arg_buf(result)
    //             .arg_scl(n as i32)

    //             .gws([n])
    //             .queue(self.device().queue().clone())
    //             .enq()?;
    //     }

    //     Ok(())
    // }

    fn sigmoid(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let n = x.shape().capacity;
        let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
        let result: &mut Memory<_> = tensor::mut_reference(result, /*on:*/ self.device())?;

        unsafe {
            ocl::Kernel::new("sigmoid_float", &self.extension_package().dependency().open_cl().program)?
                .arg_buf(x)
                .arg_buf(result)
                .arg_scl(n as i32)

                .gws([n])
                .queue(self.device().queue().clone())
                .enq()?;
        }

        Ok(())
    }

    // fn softmax(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
    //     let n = x.shape().capacity;
    //     let x: &Memory<_> = tensor::reference(x, /*on:*/ self.device())?;
    //     let result: &mut Memory<_> = tensor::mut_reference(result, /*on:*/ self.device())?;

    //     unsafe {
    //         ocl::Kernel::new("softmax_float", &self.extension_package().dependency().open_cl().program)?
    //             .arg_buf(x)
    //             .arg_buf(result)
    //             .arg_scl(n as i32)

    //             .gws([1, 1, 1])
    //             .lws([1, 1, 1])
    //             .queue(self.device().queue().clone())
    //             .enq()?;
    //     }

    //     Ok(())
    // }
}

impl<P> Extension for Context<P> where 
    P: Dependency<Package> {
    // ..
}
use parenchyma::error::Result;
use parenchyma::extension_package::Dependency;
use parenchyma::frameworks::NativeContext as Context;
use parenchyma::tensor::SharedTensor;
use super::super::{Extension, Package};
use super::super::extension_package::{Backward, Forward};

impl<P> Backward for Context<P> where 
    P: Dependency<Package> {
    fn log_softmax_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result_diff: &mut SharedTensor) -> Result {
        let x_slice = x.as_slice().unwrap();
        let x_diff_slice = x_diff.as_slice().unwrap();
        let mut sum = 0.0;
        for &grad_val in x_diff_slice.iter() {
            sum += grad_val;
        }
        let res = x_slice.iter().zip(x_diff_slice.iter())
            .map(|(x_val, x_diff_val)| {
                x_diff_val - x_val.exp() * sum
            });
        result_diff.write_iter(res)?;
        Ok(())
    }

    fn relu_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor,
        result: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {
        let res = x.as_slice().unwrap().iter()
            .zip(x_diff.as_slice().unwrap().iter())
            .map(|(x, dx)| if *x > 0.0 { *dx } else { 0.0 });
        result_diff.write_iter(res)?;
        Ok(())
    }

    fn sigmoid_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor,
        result: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {
        let res = x.as_slice().unwrap().iter().zip(x_diff.as_slice().unwrap().iter())
            .map(|(t, dt)| *t * (1.0 -*t) * *dt);
        result_diff.write_iter(res)?;
        Ok(())
    }

    fn softmax_grad(
        self: &Self,
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result_diff: &mut SharedTensor) -> Result {
        let mut dot = 0.0;
        let sig_data_slice = x.as_slice().unwrap();
        let sig_dx_slice = x_diff.as_slice().unwrap();
        for (t, dt) in sig_data_slice.iter().zip(sig_dx_slice.iter()) {
            dot += t * dt;
        }
        let res = sig_data_slice.iter().zip(sig_dx_slice.iter()).map(|(t, dt)| t * (dt - dot));
        result_diff.write_iter(res)?;
        Ok(())
    }

    fn tanh_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {
        let res = x.as_slice().unwrap().iter()
            .zip(x_diff.as_slice().unwrap().iter())
            .map(|(x, dx)| (1.0 - x.powi(2)) * *dx);
        result_diff.write_iter(res)?;
        Ok(())
    }
}

impl<P> Forward for Context<P> where 
    P: Dependency<Package> {
    fn log_softmax(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let mut max_input = ::std::f32::NEG_INFINITY;
        for &input in x.as_slice().unwrap() {
            max_input = max_input.max(input);
        }
        let mut logsum = 0.;
        for exp in x.as_slice().unwrap().iter().map(|t| (-(max_input - t)).exp()) {
            logsum += exp;
        }
        logsum = max_input + logsum.ln();
        let res = x.as_slice().unwrap().iter().map(|t| t - logsum);
        result.write_iter(res)?;
        Ok(())
    }

    fn relu(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let res = x.as_slice().unwrap().iter().map(|elem| elem.max(0.0));
        result.write_iter(res)?;
        Ok(())
    }

    fn sigmoid(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let res = x.as_slice().unwrap().iter().map(|x| 1.0 / (1.0 + (-*x).exp()));
        result.write_iter(res)?;
        Ok(())
    }

    fn softmax(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let mut exps = Vec::with_capacity(x.shape().capacity());
        let mut sum = 0.0;
        for exp in x.as_slice().unwrap().iter().map(|t| t.exp()) {
            exps.push(exp);
            sum += exp;
        }
        let res = exps.iter().map(|t| t / sum);
        result.write_iter(res)?;
        Ok(())
    }

    fn tanh(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        let res = x.as_slice().unwrap().iter().map(|elem| elem.tanh());
        result.write_iter(res)?;
        Ok(())
    }
}

impl<P> Extension for Context<P> where 
    P: Dependency<Package> {
    // ..
}
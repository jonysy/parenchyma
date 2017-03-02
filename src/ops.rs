use super::{Backend, SharedTensor, Result};

pub fn sigmoid(backend: &Backend, a: &SharedTensor<f32>, result: &SharedTensor<f32>) -> Result {

    let ref k = backend.context().kernels()["sigmoid_f32"];

    k.set_arg(0, a)?;
    k.set_arg(1, result)?;

    // TODO
    let event = backend.device().queue().enqueue_nd_range_kernel(k, &a.dims(), &[], &[])?;

    Ok(())
}
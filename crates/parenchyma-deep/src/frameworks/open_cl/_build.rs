use package::ParenchymaDeep;
use parenchyma::{Build, Result};
use parenchyma::opencl::OpenCLContext;
use parenchyma::utility::Uninitialized;
use super::Package;

impl Build<OpenCLContext<Uninitialized>> for ParenchymaDeep {

    fn build(cx: &mut OpenCLContext<Uninitialized>) -> Result<ParenchymaDeep> {

        let program = cx.create_program(&[
            include_str!("source/activation.cl"),
            include_str!("source/activationBackward.cl"),
            include_str!("source/convolution.cl")
        ])?;

        let cl_package = Package {
            tanh: program.create_kernel("tanh_float")?,
            sigmoid: program.create_kernel("sigmoid_float")?,
            relu: program.create_kernel("relu_float")?,
            elu: program.create_kernel("elu_float")?,

            tanh_backward: program.create_kernel("tanh_backward_float")?,
            sigmoid_backward: program.create_kernel("sigmoid_backward_float")?,
            relu_backward: program.create_kernel("relu_backward_float")?,
            elu_backward: program.create_kernel("elu_backward_float")?,

            convolution: program.create_kernel("convolve_ints")?,

            program,
        };

        Ok(ParenchymaDeep { cl: cl_package })
    }
}
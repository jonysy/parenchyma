use ocl;
use std::ffi::CString;
use parenchyma::error::Result;
use parenchyma::frameworks::OpenCLContext;

/// Caches instances of `Kernel`
#[derive(Debug)]
pub struct OpenCLPackage {
    pub(in frameworks::open_cl) program: ocl::Program,
}

impl OpenCLPackage {
    pub fn compile(cx: &mut OpenCLContext<()>) -> Result<OpenCLPackage> {
        let program = cx.program(vec![
            CString::new(include_str!("source/activation.cl")).unwrap(),
            CString::new(include_str!("source/activationBackward.cl")).unwrap(),
            CString::new(include_str!("source/convolution.cl")).unwrap(),
            CString::new(include_str!("source/softmax.cl")).unwrap()
        ])?;

        // let cl_package = Package {
        //     tanh: program.create_kernel("tanh_float")?,
        //     sigmoid: program.create_kernel("sigmoid_float")?,
        //     relu: program.create_kernel("relu_float")?,
        //     elu: program.create_kernel("elu_float")?,

        //     tanh_backward: program.create_kernel("tanh_backward_float")?,
        //     sigmoid_backward: program.create_kernel("sigmoid_backward_float")?,
        //     relu_backward: program.create_kernel("relu_backward_float")?,
        //     elu_backward: program.create_kernel("elu_backward_float")?,

        //     convolution: program.create_kernel("convolve_ints")?,

        //     program,
        // };

        Ok(OpenCLPackage { program })
    }
}
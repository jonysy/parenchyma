use super::Result;
use super::api;

#[derive(Clone, Debug)]
pub struct OpenCLKernels {
    sigmoid: api::Kernel,
}

impl OpenCLKernels {

    pub fn new(program: &api::Program) -> Result<OpenCLKernels> {
        Ok(OpenCLKernels {
            sigmoid: program.create_kernel("array_sigmoid_f32")?,
        })
    }
}
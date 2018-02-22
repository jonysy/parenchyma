mod build;

use extension::{ActivationMode, Backward, Deep, Forward};
use package::ParenchymaDeep;

use parenchyma::{Result, SharedTensor};
use parenchyma::opencl::OpenCLContext;
use parenchyma::opencl::high;
use parenchyma::utility::Uninitialized;

#[derive(Debug)]
pub struct Package {
    program: high::Program,

    // === activation

    tanh: high::Kernel,
    sigmoid: high::Kernel,
    relu: high::Kernel,
    elu: high::Kernel,

    // === activation backward

    tanh_backward: high::Kernel,
    sigmoid_backward: high::Kernel,
    relu_backward: high::Kernel,
    elu_backward: high::Kernel,

    // == conv
    convolution: high::Kernel,
}

impl Deep for OpenCLContext<ParenchymaDeep> { }

impl Forward for OpenCLContext<ParenchymaDeep> {

    fn activation(
        &self, 
        mode: ActivationMode, 
        input: &SharedTensor, 
        output: &mut SharedTensor) -> Result {

        use extension::ActivationMode::*;

        let kernel = match mode {
            Tanh => unsafe { &self.package().cl.tanh },
            Sigmoid => unsafe { &self.package().cl.sigmoid },
            ReLu => unsafe { &self.package().cl.relu },
            Elu => unsafe { &self.package().cl.elu },
        };

        let length = input.shape.capacity();

        kernel.set_arg(0, input.read(self)?)?;
        kernel.set_arg(1, output.write(self)?)?;
        kernel.set_arg(2, &length)?;

        let global_work = &[length];
        let local_work = &[];

        // TODO event_wait_list
        let events = &[];

        // TODO
        let event = self.device().queue()
            .enqueue_nd_range_kernel(kernel, global_work, local_work, events)?;

        Ok(())
    }
}

impl Backward for OpenCLContext<ParenchymaDeep> {

    fn activation_backward(
        &self, 
        mode: ActivationMode, 
        input: &SharedTensor, 
        input_diff: &SharedTensor, 
        output_diff: &mut SharedTensor) -> Result {

        use extension::ActivationMode::*;

        let kernel = match mode {
            Tanh => unsafe { &self.package().cl.tanh_backward },
            Sigmoid => unsafe { &self.package().cl.sigmoid_backward },
            ReLu => unsafe { &self.package().cl.relu_backward },
            Elu => unsafe { &self.package().cl.elu_backward },
        };

        let length = input.shape.capacity();

        kernel.set_arg(0, input.read(self)?)?;
        kernel.set_arg(1, input_diff.read(self)?)?;
        kernel.set_arg(2, output_diff.write(self)?)?;
        kernel.set_arg(3, &length)?;

        let global_work = &[length];
        let local_work = &[];

        // TODO event_wait_list
        let events = &[];


        // TODO
        let event = self.device().queue()
            .enqueue_nd_range_kernel(kernel, global_work, local_work, events)?;

        Ok(())
    }
}
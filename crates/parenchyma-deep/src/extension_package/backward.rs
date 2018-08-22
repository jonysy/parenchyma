use parenchyma::error::Result;
use parenchyma::prelude::SharedTensor;
use super::{ConvolutionConfiguration, LrnConfiguration, PoolingConfiguration};

pub trait Backward {
    /// Computes the gradient of a [CNN convolution] over the input tensor `x` with respect 
    /// to the data.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// [CNN convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    fn convolution_grad_data(
        self: &Self,
        filter: &SharedTensor, 
        x_diff: &SharedTensor,
        result_diff: &mut SharedTensor, 
        workspace: &mut SharedTensor<u8>, 
        configuration: &ConvolutionConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of a [CNN convolution][convolution] with respect to the filter.
    ///
    /// Saves the result to `filter_diff`.
    ///
    /// [CNN convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    fn convolution_grad_filter(
        self: &Self, 
        src_data: &SharedTensor, 
        dest_diff: &SharedTensor, 
        filter_diff: &mut SharedTensor, 
        workspace: &mut SharedTensor<u8>, 
        configuration: &ConvolutionConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of a logarithmic softmax over the input tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    fn log_softmax_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of a [LRN][lrn] over the input Tensor `x` with complete memory management.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `lrn_grad_plain`.
    fn lrn_grad(
        self: &Self,
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result: &SharedTensor, 
        result_diff: &mut SharedTensor, 
        configuration: &LrnConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of [max pooling] over the input Tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// [max pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    fn pooling_max_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result: &SharedTensor, 
        result_diff: &mut SharedTensor, 
        configuration: &PoolingConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of [ReLU] over the input tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// [ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    fn relu_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor,
        result: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of [ReLU] over the input tensor `x`.
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// [ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    fn relu_pointwise_grad(&self, x: &SharedTensor, x_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of a [sigmoid function] over the input tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor,
        result: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of a [sigmoid function] over the input tensor `x`.
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid_pointwise_grad(&self, x: &SharedTensor, x_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of a [softmax] over the input tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    fn softmax_grad(
        self: &Self,
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of [tanh] over the input Tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    fn tanh_grad(
        self: &Self, 
        x: &SharedTensor, 
        x_diff: &SharedTensor, 
        result: &SharedTensor,
        result_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the gradient of [tanh] over the input Tensor `x`.
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    fn tanh_pointwise_grad(&self, x: &SharedTensor, x_diff: &mut SharedTensor) -> Result {
        unimplemented!()
    }
}
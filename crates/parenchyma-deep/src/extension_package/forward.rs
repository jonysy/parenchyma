use parenchyma::error::Result;
use parenchyma::prelude::SharedTensor;
use super::{ConvolutionConfiguration, LrnConfiguration, PoolingConfiguration};

pub trait Forward {
    /// Computes a [CNN convolution] over the input tensor `x`, and then saves the `result`.
    ///
    /// [CNN convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    fn convolution(
        self: &Self, 
        filter: &SharedTensor, 
        x: &SharedTensor, 
        result: &mut SharedTensor,
        workspace: &mut SharedTensor<u8>,
        configuration: &ConvolutionConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes the exponential linear unit [new] over tensor `x`.
    ///
    /// Saves the `result`.
    fn elu(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes a logarithmic softmax over the input tensor `x`, and then saves the `result`.
    fn log_softmax(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes a [local response normalization] over the input tensor `x`.
    ///
    /// Saves the result to `result`.
    ///
    /// [local response normalization]: https://en.wikipedia.org/wiki/lrnal_neural_network
    fn lrn(
        self: &Self, 
        x: &SharedTensor, 
        result: &mut SharedTensor, 
        configuration: &LrnConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes non-linear down-sampling ([max pooling]) over the input tensor `x`.
    ///
    /// Saves the result to `result`.
    ///
    /// [max pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    fn pooling_max(
        self: &Self, 
        x: &SharedTensor, 
        result: &mut SharedTensor, 
        configuration: &PoolingConfiguration) -> Result {
        unimplemented!()
    }
    /// Computes the [rectified linear units] over tensor `x`.
    ///
    /// Saves the `result`.
    ///
    /// [rectified linear units]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    fn relu(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the [rectified linear units] over the input Tensor `x`.
    ///
    /// note: pointwise operations overwrite the input with the result of the operation.
    ///
    /// [rectified linear units]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    fn relu_pointwise(&self, x: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the [sigmoid function] over tensor `x`.
    ///
    /// Saves the `result`.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the [sigmoid function][sigmoid] over the input tensor `x`.
    ///
    /// note: pointwise operations overwrite the input with the result of the operation.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid_pointwise(&self, x: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes a [softmax] over the input tensor `x`.
    ///
    /// Saves the result to `result`.
    ///
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    fn softmax(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the [hyperbolic tangent] over tensor `x`.
    ///
    /// Saves the `result`.
    ///
    /// [hyperbolic tangent]: https://en.wikipedia.org/wiki/Hyperbolic_function
    fn tanh(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {
        unimplemented!()
    }
    /// Computes the [hyperbolic tangent][tanh] over the input Tensor `x`.
    ///
    /// note: pointwise operations overwrite the input with the result of the operation.
    ///
    /// [hyperbolic tangent]: https://en.wikipedia.org/wiki/Hyperbolic_function
    fn tanh_pointwise(&self, x: &mut SharedTensor) -> Result {
        unimplemented!()
    }
}
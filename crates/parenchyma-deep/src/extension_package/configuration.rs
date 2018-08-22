#[derive(Clone, Copy, Debug)]
pub struct ConvolutionConfiguration;

// impl ConvolutionConfiguration {
//     /// Creates a new convolution configuration, which needs to be passed to further 
//     /// convolution operations.
//     pub fn new<P>(
//         backend: &Backend<P>, 
//         src: &SharedTensor, 
//         dest: &SharedTensor,
//         filter: &mut SharedTensor,
//         algo_forward: ConvForwardAlgo,
//         algo_backward_filter: ConvBackwardDataAlgo,
//         algo_backward_data: ConvBackwardDataAlgo,
//         stride: &[i32],
//         zero_padding: &[i32]) -> Result<ConvolutionConfiguration> {

//         unimplemented!()
//     }
// }

#[derive(Clone, Copy, Debug)]
pub struct LrnConfiguration;

#[derive(Clone, Copy, Debug)]
pub struct PoolingConfiguration;
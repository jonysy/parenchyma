use cuda::CudaDeviceHandle;
use parenchyma::Processor;

#[derive(Clone, Debug)]
pub struct Device {
    pub(super) handle: CudaDeviceHandle,
    pub name: String,
    pub multiprocessors: i32,
    pub processor: Processor,
}
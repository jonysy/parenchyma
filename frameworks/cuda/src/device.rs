use api::DeviceHandle;
use parenchyma::Processor;

#[derive(Clone, Debug)]
pub struct CudaDevice {
    pub(super) handle: DeviceHandle,
    pub name: String,
    pub multiprocessors: i32,
    pub processor: Processor,
}
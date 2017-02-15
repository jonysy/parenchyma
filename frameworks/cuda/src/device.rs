use api::DeviceHandle;
use parenchyma::Category;

#[derive(Clone, Debug)]
pub struct CudaDevice {
    pub(super) handle: DeviceHandle,
    pub name: String,
    pub multiprocessors: i32,
    pub category: Category,
}
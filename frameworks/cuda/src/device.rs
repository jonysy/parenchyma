use parenchyma::Processor;

#[derive(Clone, Debug)]
pub struct Device {
    // id: core::DeviceId,
    pub processor: Processor,
}
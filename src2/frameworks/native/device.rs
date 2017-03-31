use error::Result;
use framework::Framework;
use hardware::{Hardware, HardwareDevice, HardwareType};
use memory::{Alloc, BoxChunk, FlatBox};
use super::{Native, NativeMemory};

pub static HOST: NativeDevice = NativeDevice;

lazy_static! {
    pub static ref HARDWARE: Vec<Hardware> = vec![Hardware {
        id: 0,
        framework: Native::ID,
        processor: HardwareType::CPU,
        name: String::from("HOST CPU"),
        compute_units: 1,
    }];
}

/// The native/host CPU device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDevice;

impl HardwareDevice for NativeDevice {

    fn prealloc(&self, capacity: usize) -> Result<BoxChunk> {
        let chunk = (HOST, NativeMemory(FlatBox::new(capacity)));
        let boxed_chunk = Box::new(chunk);
        Ok(boxed_chunk)
    }
}

impl Alloc<f32> for NativeDevice {
    fn alloc_place(&self, data: Vec<f32>) -> Result<BoxChunk> {

        let chunk = (HOST, NativeMemory(FlatBox::from(data.into_boxed_slice())));
        let boxed_chunk = Box::new(chunk);

        Ok(boxed_chunk)
    }
}

impl Alloc<f64> for NativeDevice { }
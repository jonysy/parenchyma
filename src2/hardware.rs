use error::Result;
use memory::{Alloc, BoxChunk};
use std::any::{Any, TypeId};

/// General categories for devices, used to identify the type of a device.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HardwareType {
    /// Used for accelerators. Accelerators can communicate with host processor using a peripheral
    /// interconnect such as PCIe.
    Accelerator,
    /// Used for devices that are host processors. The host processor runs the implementations
    /// and is a single or multi-core CPU.
    CPU,
    /// Used for GPU devices.
    GPU,
    /// Used for anything else.
    Other,
}

/// Hardware can be GPUs, multi-core CPUs or DSPs, Cell/B.E. processor or whatever else
/// is supported by the provided framework. The struct holds all important information about 
/// the hardware. To execute code on hardware, turn hardware into a [`Device`].
///
/// [`Device`]: [device]: ./struct.Device.html
#[derive(Clone, Debug)]
pub struct Hardware {
    /// The unique ID of the hardware.
    pub id: usize,
    /// Framework marker
    pub framework: &'static str,
    /// The type of compute device, such as a CPU or a GPU.
    pub processor: HardwareType,
    /// The name.
    pub name: String,
    /// The number of compute units.
    ///
    /// A compute device usually has multiple compute units.
    pub compute_units: usize,
}

/// An device capable of processing data.
///
/// A compute device is a processor, such as a CPU or a GPU. `Device` is simply 
/// an [alias][issue#8634] for the various trait bounds associated with a compute device.
///
/// [issue#1733]: https://github.com/rust-lang/rfcs/pull/1733
pub trait HardwareDevice: Alloc<f32> + Alloc<f64> + Any {

    /// Preallocates memory on the device.
    fn prealloc(&self, byte_size: usize) -> Result<BoxChunk> { unimplemented!() }
}

impl HardwareDevice {

    #[inline]
    pub fn is<T: HardwareDevice>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    pub fn downcast_ref<T: HardwareDevice>(&self) -> Option<&T> {
        if self.is::<T>() {
            unsafe {
                Some(&*(self as *const HardwareDevice as *const T))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn downcast_mut<T: HardwareDevice>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            unsafe {
                Some(&mut *(self as *mut HardwareDevice as *mut T))
            }
        } else {
            None
        }
    }
}
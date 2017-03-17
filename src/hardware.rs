use frameworks::native::NativeDevice;
use frameworks::opencl::OpenCLDevice;
use super::{Memory, Result, Shape};

/// An device capable of processing data.
///
/// A compute device is a processor, such as a CPU or a GPU. `Device` is simply 
/// an [alias][issue#8634] for the various trait bounds associated with a compute device.
///
/// [issue#8634]: https://github.com/rust-lang/rust/issues/8634
pub trait Device: 
    'static 
    + Alloc<u8> + Alloc<f32> + Alloc<f64> 
    + Synch<u8> + Synch<f32> + Synch<f64> 
    + Viewable {

    }

impl<D> Device for D where D: 
    'static 
    + Alloc<u8> + Alloc<f32> + Alloc<f64> 
    + Synch<u8> + Synch<f32> + Synch<f64> 
    + Viewable {
        
    }

/// A _viewable_ device.
pub trait Viewable {

    /// Returns a device _view_.
    fn view(&self) -> ComputeDevice;
}

/// A wrapper around the various compute devices.
///
/// `ComputeDevice` and `Viewable` use the [_family_ pattern][pattern].
///
/// [pattern]: https://www.reddit.com/r/rust/comments/2rdoxx/enum_variants_as_types/cnezl0p/
#[derive(Debug, Eq, PartialEq)]
pub enum ComputeDevice {
    /// A native host device
    Native(NativeDevice),
    /// An OpenCL device
    OpenCL(OpenCLDevice),
}

impl ComputeDevice {

    /// Returns the device
    pub fn device(&self) -> &Device {
        match *self {
            ComputeDevice::Native(ref d) => d,
            ComputeDevice::OpenCL(ref d) => d,
        }
    }
}

/// Allocator
pub trait Alloc<T> {

    /// Allocates memory on the device.
    fn alloc(&self, shape: &Shape) -> Result<Memory<T>>;

    /// Allocates and transfers memory `data` to the device.
    fn allocwrite(&self, shape: &Shape, data: Vec<T>) -> Result<Memory<T>>;
}

/// Synchronizer
///
/// note: host <-> GPU for now.. GPU <-> GPU later..
pub trait Synch<T> {

    // TODO refactor

    /// Synchronizes `memory` from `source`.
    fn write(&self, memory: &mut Memory<T>, s_location: &ComputeDevice, s: &Memory<T>) -> Result;

    /// Synchronizes `memory` to `destination`.
    fn read(&self, memory: &Memory<T>, d_location: &mut ComputeDevice, d: &mut Memory<T>) -> Result;
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
    pub kind: HardwareKind,
    /// The name.
    pub name: String,
    /// The number of compute units.
    ///
    /// A compute device usually has multiple compute units.
    pub compute_units: usize,
}

/// General categories for devices, used to identify the type of a device.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HardwareKind {
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
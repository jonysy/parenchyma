use std::fmt::Debug;

use super::{Memory, Shape};
use super::error::Result;

/// Represents the location of a buffer or memory, which the associated device can 
/// use to access it.
#[derive(Debug, Eq, PartialEq)]
pub struct Address {
    /// A string literal containing the name of the framework.
    pub framework: &'static str,
    /// The context identifier
    pub context: isize,
    /// The device identifier.
    pub device: isize,
}

/// A device capable of processing data.
///
/// The `T` type associated with the [`SharedTensor`](./struct.SharedTensor.html).
pub trait ComputeDevice<T> {

    /// Allocates memory on the device.
    fn allocate(&self, shape: &Shape) -> Result<Memory<T>>;

    /// Allocates memory on the device.
    fn allocate_with(&self, shape: &Shape, slice: &mut [T]) -> Result<Memory<T>>;

    // /// Synchronizes `memory` from `source`.
    // fn sync_in(&self, memory: &mut Memory<T>, source: &Memory<T>) -> Result;

    /// Synchronizes `memory` to `destination`.
    fn sync_out(&self, memory: &Memory<T>, destination: &mut Memory<T>) -> Result;

    /// Returns the location of the device.
    ///
    /// The `addr` method is used by `SharedTensor`s for memory storage purposes. The _address_
    /// is simply the name of the framework associated with the device, the device's unique
    /// identifier, and an integer associated with the context the device is contained in.
    fn addr(&self) -> Address;
}

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
pub trait Context: Debug {

}

/// A trait implemented for all frameworks. `Framework`s contain a list of all available devices as 
/// well as other objects specific to the implementor.
///
/// The default framework is simply the host CPU for common computation. To make use of other
/// devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to access the 
/// processing capabilities of the device(s).
pub trait Framework: Debug {
    /// The name of the framework.
    ///
    /// This associated constant is mainly used for the purposes of debugging and reporting errors.
    ///
    /// note: *uses the "SCREAMING_SNAKE_CASE" naming convention (e.g., `"OPEN_CL"`).
    const FRAMEWORK_NAME: &'static str;

    // type Context: Context;

    // fn try_init(&self) -> Result;

    // fn try_context(&self, selection: Vec<Hardware>) -> Result<Box<Context>>;

    // TODO:
    // https://github.com/rust-lang/rust/issues/29924
    #[doc(hidden)]
    fn name(&self) -> &'static str {
        Self::FRAMEWORK_NAME
    }
}

// /// The object-safe version of `Framework`.
// trait FrameworkObject: Debug { }

/// The generic hardware representation for a `ComputeDevice`.
///
/// A compute device is a processor, such as a CPU or a GPU.
pub struct Hardware {
    /// The unique ID of the hardware.
    id: usize,
    /// The type of compute device, such as a CPU or a GPU.
    kind: HardwareKind,
    /// The name.
    name: String,
    /// The number of compute units.
    ///
    /// A compute device usually has multiple compute units.
    compute_units: usize,
    // /// Framework marker
    // framework: PhantomData<F>,
}

/// General categories for devices, used to identify the type of a device.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HardwareKind {
    /// Used for accelerators. Accelerators can communicate with host processor using a peripheral
    /// interconnect such as PCIe.
    Accelerator,
    /// Used for devices that are host processors. The host processor runs the implementations
    /// and is a single or multi-core CPU.
    Central,
    /// Used for GPU devices.
    Graphics,
    /// Used for anything else.
    Other,
}
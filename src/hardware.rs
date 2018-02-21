//! Hardware can be GPUs, multi-core CPUs or DSPs, Cell/B.E. processor or whatever else
//! is supported by the provided framework. The struct holds all important information about 
//! the hardware. To execute code on hardware, turn hardware into a [`ComputeDevice`].
//!
//! [`Device`]: [device]: ./compute_device/struct.Device.html

/// Representation for hardware across frameworks.
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
    /// A compute unit is the fundamental unit of computation. A compute device usually has 
    /// multiple compute units.
    pub compute_units: usize,
}

/// General classes for devices, used to identify the type of a device.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HardwareKind {
    /// Used for accelerators. Accelerators can communicate with host processor using a peripheral
    /// interconnect such as PCIe.
    Accelerator,
    /// Used for cells.
    Cell,
    /// Used for devices that are host processors. The host processor runs the implementations
    /// and is a single or multi-core CPU.
    CPU,
    /// Used for digital signal processors.
    DSP,
    /// Used for GPU devices.
    GPU,
    /// Used for anything else.
    Unknown,
}
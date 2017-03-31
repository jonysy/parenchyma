/// General categories for hardware, used to identify the type of a device.
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
/// is supported by the provided framework. The `Hardware` struct holds all the important 
/// information about the actual bare metal underlying hardware. To execute code on hardware, turn 
/// hardware into a [`Device`].
///
/// [`Device`]: [device]: ./struct.Device.html
#[derive(Clone, Debug)]
pub struct Hardware {
    /// The unique ID of the hardware.
    pub id: usize,
    /// The name.
    pub name: String,
    /// The type of processor, such as a CPU, a GPU, a DSP, or any other processor provided and
    /// supported by the framework implementation.
    pub processor: HardwareType,
    /// The number of compute units.
    ///
    /// A compute device usually has multiple compute units.
    pub compute_units: usize,
}
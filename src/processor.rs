use std::borrow::Cow;

/// Used to identify the type of a device.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Processor {
    /// Used for accelerators. Accelerators can communicate with host processor using a peripheral
    /// interconnect such as PCIe.
    Accelerator,
    /// Used for devices that are host processors. The host processor runs the implementations
    /// and is a single or multi-core CPU.
    Central,
    /// Used for GPU devices.
    Graphics,
    /// Used for anything else.
    Custom(Cow<'static, str>),
}
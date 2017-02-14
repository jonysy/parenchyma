use std::borrow::Cow;

/// General categories for devices.
#[derive(Debug, Clone)]
pub enum Category {
    /// Accelerators
	Accelerator,
    /// CPU devices (host processors)
	Cpu,
    /// GPU devices
	Gpu,
    /// Used for anything else
	Other(Cow<'static, str>),
}
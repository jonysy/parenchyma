use std::borrow::Cow;

/// The available processors.
#[derive(Debug, Clone)]
pub enum Processor {
	Accelerator,
	Cpu,
	Gpu,
	Other(Cow<'static, str>),
}
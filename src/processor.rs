use std::borrow::Cow;

/// The available processors.
#[derive(Clone)]
pub enum Processor {
	Cpu,
	Gpu,
	Accelerator,
	Other(Cow<'static, str>),
}
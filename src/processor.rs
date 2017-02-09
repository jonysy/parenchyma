use std::borrow::Cow;

#[derive(Debug, Clone)]
pub enum Processor {
	Accelerator,
	Cpu,
	Gpu,
	Other(Cow<'static, str>),
}
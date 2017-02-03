use std::borrow::Cow;

#[derive(Clone)]
pub enum Processor {
	Cpu,
	Gpu,
	Accelerator,
	Other(Cow<'static, str>),
}
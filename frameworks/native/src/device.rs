use parenchyma::Processor;
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub struct NativeDevice {
	pub(super) name: Cow<'static, str>,
	pub(super) compute_units: isize,
	pub(super) processor: Processor,
}
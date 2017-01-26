use std::borrow::Cow;
use std::marker::PhantomData;
use super::Processor;

#[derive(Clone)]
pub struct Device<T> {
	pub(super::super) id: isize,
	pub(super::super) name: Option<Cow<'static, str>>,
	pub(super::super) processor: Option<Processor>,
	pub(super::super) compute_units: Option<isize>,
	pub(super::super) phantom: PhantomData<T>
}

impl<T> Default for Device<T> {

	fn default() -> Device<T> {

		Device {
			id: -1,
			name: None,
			processor: None,
			compute_units: None,
			phantom: PhantomData,
		}
	}
}
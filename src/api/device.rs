use std::borrow::Cow;
use std::marker::PhantomData;
use super::Processor;

#[derive(Clone)]
pub struct Device<T> {
	pub id: isize,
	pub name: Option<Cow<'static, str>>,
	pub processor: Option<Processor>,
	pub compute_units: Option<isize>,
	pub phantom: PhantomData<T>
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
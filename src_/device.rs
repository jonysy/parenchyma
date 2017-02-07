use std::borrow::Cow;
use std::marker::PhantomData;
use super::Processor;

#[derive(Debug, Clone)]
pub struct Device<F> {
	//pub implementation_id: Option<isize>,
	pub id: isize,
	pub compute_units: Option<isize>,
	pub name: Option<Cow<'static, str>>,
	pub processor: Option<Processor>,
	pub phantom: PhantomData<F>,
}

impl<F> Default for Device<F> {

	fn default() -> Self {

		Device {
			id: -1,
			compute_units: None,
			name: None,
			processor: None,
			phantom: PhantomData,
		}
	}
}
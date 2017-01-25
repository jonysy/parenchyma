pub use self::kind::DeviceKind;
mod kind;

use std::borrow::Cow;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Device<T> {
	pub(super) id: isize,
	pub(super) name: Option<Cow<'static, str>>,
	pub(super) kind: Option<DeviceKind>,
	pub(super) compute_units: Option<isize>,
	pub(super) phantom: PhantomData<T>
}

impl<T> Default for Device<T> {

	fn default() -> Device<T> {

		Device {
			id: -1,
			name: None,
			kind: None,
			compute_units: None,
			phantom: PhantomData,
		}
	}
}
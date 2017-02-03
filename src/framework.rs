use std::convert::TryFrom;
use super::{Context, Device, Error};

pub trait Framework: Sized {

	type Context: Context + TryFrom<Vec<Device<Self>>, Err = Error>;

	fn new() -> Self;

	fn devices(&self) -> &[Device<Self>];
}
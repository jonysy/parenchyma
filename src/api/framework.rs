use error::Error;
use super::{Context, Device};

pub trait Framework: Sized {
	
	fn new() -> Self;

	fn devices(&self) -> &[Device<Self>];

	fn try_context(&self, devices: Vec<Device<Self>>) -> Result<Context, Error>;
}
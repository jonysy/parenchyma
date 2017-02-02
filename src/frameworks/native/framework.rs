use api::{Context, Device, Framework, Processor};
use error::Error;
use std::borrow::Cow;
use std::marker::PhantomData;
use super::{NativeContext, NativeDevice};

#[derive(Clone)]
pub struct Native {
	devices: Vec<NativeDevice>,
}

impl Framework for Native {
	
	fn new() -> Self {
		let device = NativeDevice {
			id: 1,
			name: Some(Cow::from("Host CPU")),
			processor: Some(Processor::Cpu),
			compute_units: Some(1),
			phantom: PhantomData,
		};

		Native { devices: vec![device] }
	}

	fn devices(&self) -> &[NativeDevice] {

		&self.devices
	}

	fn try_context(&self, devices: Vec<Device<Self>>) -> Result<Context, Error> {

		Ok(Context::Native(NativeContext { devices: devices }))
	}
}
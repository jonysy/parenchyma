use parenchyma::{Framework, Processor};
use std::borrow::Cow;
use std::marker::PhantomData;
use super::{NativeContext, NativeDevice};

#[derive(Clone)]
pub struct Native {
	devices: Vec<NativeDevice>,
}

impl Framework for Native {
	type Context = NativeContext;

	const ID: &'static str = "NATIVE";

	fn new() -> Native {
		let device = NativeDevice {
			id: 1,
			compute_units: Some(1),
			name: Some(Cow::from("Host CPU")),
			processor: Some(Processor::Cpu),
			phantom: PhantomData,
		};

		Native { devices: vec![device] }
	}

	fn devices(&self) -> &[NativeDevice] {

		&self.devices
	}
}
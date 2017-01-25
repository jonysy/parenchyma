pub use self::context::NativeContext;
pub use self::device::NativeDevice;
pub use self::memory::NativeMemory;
#[cfg(not(feature = "unstable_alloc"))]
pub use self::stable_alloc::allocate_boxed_slice;
#[cfg(feature = "unstable_alloc")]
pub use self::unstable_alloc::allocate_boxed_slice;

mod context;
mod device;
mod memory;
#[cfg(not(feature = "unstable_alloc"))]
mod stable_alloc;
#[cfg(feature = "unstable_alloc")]
mod unstable_alloc;

use {ContextImp, DeviceKind, Error, Framework};
use std::borrow::Cow;
use std::marker::PhantomData;

pub struct Native {
	devices: Vec<NativeDevice>,
}

impl Framework for Native {

	type Context = NativeContext;

	const NAME: &'static str = "NATIVE";

	fn new() -> Self {

		let cpu = NativeDevice {
			id: 1,
			name: Some(Cow::from("Host CPU")),
			kind: Some(DeviceKind::Cpu),
			compute_units: Some(1),
			phantom: PhantomData,
		};

		Native { devices: vec![cpu] }
	}

	fn devices(&self) -> &[NativeDevice] {

		&self.devices
	}

	fn new_context(&self, devices: &[NativeDevice]) -> Result<ContextImp, Error> {

		Ok(ContextImp::Native(
			NativeContext {
				devices: devices.to_vec() 
			}
		))
	}
}
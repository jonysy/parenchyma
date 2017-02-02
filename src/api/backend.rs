use api::{Context, Device, Framework};
use error::Error;

/// `Backend` is the heart of Parenchyma. `Backend` provides the interface for running parallel 
/// computations on one ore many devices.
///
/// This is the abstraction over which you are interacting with your devices. You can create a
/// `Backend` for computation by first choosing a specific `Framework` such as `OpenCL` and
/// afterwards selecting one or many available hardwares to create a `Backend`.
///
/// A `Backend` provides you with the functionality of managing the memory of the devices and copying
/// your objects from host to devices and the other way around. Additionally you can execute 
/// operations in parallel through kernel functions on the device(s) of the `Backend`.
pub struct Backend<F> {
	framework: F,
	pub context: Context,
}

impl<F> Backend<F> where F: Framework {

	pub fn new<T>(framework: F, closure: T) -> Result<Backend<F>, Error> 
		where T: for<'frwk> Fn(&'frwk [Device<F>]) -> Vec<Device<F>>
	{
		let devices = closure(framework.devices());

		let context = framework.try_context(devices)?;

		Ok(Backend {
			framework: framework,
			context: context,
		})
	}
}
use std::convert::TryFrom;
use super::{Device, Error, Framework};

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
pub struct Backend<F> where F: Framework {
	context: F::Context,
	framework: F,
}

impl<F> Backend<F> where F: Framework {

	/// # Example
	///
	/// ```rust,ignore
	/// extern crate parenchyma;
	/// extern crate parenchyma_native;
	///
	/// use parenchyma::{Backend, Framework};
	/// use parenchyma_native::Native;
	///
	/// 
	///	// Construct a new framework.
	///	let framework = Native::new();
	///
	///	// Available devices can be obtained through the framework.
	///	let selection = framework.devices().to_vec();
	///
	///	// Create a ready to go `Backend` from the framework.
	///	let backend = Backend::new(framework, selection).expect("Something went wrong!");
	/// ```
	pub fn new(framework: F, selection: Vec<Device<F>>) -> Result<Backend<F>, Error> {

		let context = F::Context::try_from(selection)?;
		let backend = Backend { framework: framework, context: context};

		Ok(backend)
	}

	/// # Example
	///
	/// ```rust,ignore
	/// use parenchyma::Backend;
	/// use parenchyma_native::Native;
	///
	/// let backend = Backend::<Native>::default().expect("Something went wrong!");
	/// ```
	pub fn default() -> Result<Backend<F>, Error> where F: Clone {
		let framework = F::new();
		let selection = framework.devices().to_vec();
		Backend::new(framework, selection)
	}

	pub fn context(&self) -> &F::Context {
		&self.context
	}

	pub fn framework(&self) -> &F {
		&self.framework
	}
}
use super::{Context, Device};

/// Base functionality for all frameworks.
///
/// The default Framework would be a plain host CPU for common computation. To make use of other
/// computation hardware such as GPUs you would choose other computation Frameworks such as OpenCL
/// or CUDA, which provide the access to those hardwares for computation.
///
/// To start backend-agnostic and highly parallel computation, you start by initializing a
/// `Framework` implementation, resulting in an initialized framework that contains a list of
/// all available devices through that framework, as well as other things specific to the framework.
pub trait Framework: Sized {
	/// The `Context` representation for this framework.
	type Context: Context<Framework = Self>;

	/// The name of the framework (e.g., `"OPEN_CL"`).
	const ID: &'static str;

	/// Initializes a new framework.
	fn new() -> Self;

	/// Returns the cached and available devices.
	fn devices(&self) -> &[Device<Self>];
}
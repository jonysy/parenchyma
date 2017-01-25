use {Context, ContextImp, Device, Error};

pub trait Framework {

	type Context: Context;

	const NAME: &'static str;

	fn new() -> Self;

	fn devices(&self) -> &[Device<Self::Context>];

	// /// The initialized binary
	// fn binary(&self) -> &Binary { .. }

	fn new_context(&self, devices: &[Device<Self::Context>]) -> Result<ContextImp, Error>;
}
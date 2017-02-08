use parenchyma::Processor;

#[derive(Clone, Debug)]
pub struct NativeDevice {
	pub(super) name: &'static str,
	pub(super) compute_units: isize,
	pub(super) processor: Processor,
}
#[derive(Clone)]
pub enum DeviceKind {
	Cpu,
	Gpu,
	Accelerator,
	Other,
}
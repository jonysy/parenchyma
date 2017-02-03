#![feature(pub_restricted, try_from)]

pub mod error;
pub use self::context::{Context, ObjectSafeContext};
pub use self::backend::Backend;
pub use self::device::Device;
pub use self::framework::Framework;
pub use self::processor::Processor;
pub use self::shared_tensor::{SharedTensor, Tensor};

mod context;
mod backend;
mod device;
mod framework;
mod processor;
mod shared_tensor;
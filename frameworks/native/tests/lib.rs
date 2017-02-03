#![cfg(test)]
#![feature(try_from)]

extern crate parenchyma;
extern crate parenchyma_native;

mod backend_spec {
	use parenchyma::Backend;
	use parenchyma_native::Native;

	#[test]
	fn it_can_create_default_backend() {

		assert!(Backend::<Native>::default().is_ok());
	}
}

mod framework_spec {
	use parenchyma::Framework;
	use parenchyma_native::Native;

	#[test]
	fn it_works() {
		let framework = Native::new();
		assert_eq!(framework.devices().len(), 1);
	}
}

mod shared_memory_spec {
	use parenchyma::{Framework, SharedTensor};
	use parenchyma_native::{Native, NativeContext};
	use std::convert::TryFrom;

	#[test]
	fn it_creates_new_shared_memory_for_native() {
		let native = Native::new();
		let context = NativeContext::try_from(native.devices().to_vec()).unwrap();
		let mut shared_data = SharedTensor::<f32>::new(vec![10]);
		let data = shared_data.write_only(&context).unwrap().as_slice::<f32>();
		assert_eq!(10, data.len());
	}

	#[test]
	fn it_fails_on_initialized_memory_read() {
		use parenchyma::error;

		let native = Native::new();
		let context = NativeContext::try_from(native.devices().to_vec()).unwrap();
		let mut shared_data = SharedTensor::<f32>::new(vec![10]);

		assert_eq!(
			*shared_data.read(&context).unwrap_err().category(), 
			error::Category::Memory(error::MemoryCategory::Uninitialized)
		);

		assert_eq!(
			*shared_data.read_write(&context).unwrap_err().category(),
			error::Category::Memory(error::MemoryCategory::Uninitialized)
		);

		shared_data.write_only(&context).unwrap();
		shared_data.drop_context(&context).unwrap();

		assert_eq!(
			*shared_data.read(&context).unwrap_err().category(),
			error::Category::Memory(error::MemoryCategory::Uninitialized)
        );
	}
}
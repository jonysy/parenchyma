#![cfg(test)]

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
    fn it_can_init_native_framework() {
        let framework = Native::new();
        assert_eq!(framework.devices().len(), 1);
    }
}

mod shared_memory_spec {
    use parenchyma::{Context, Framework, SharedTensor};
    use parenchyma::error::ErrorKind;
    use parenchyma_native::{Native, NativeContext};

    #[test]
    fn it_creates_new_shared_memory_for_native() {
        let native = Native::new();
        let context = NativeContext::new(native.devices().to_vec()).unwrap();
        let mut shared_data = SharedTensor::<f32>::new(vec![10]);
        let data = shared_data.write_only(&context).unwrap().as_slice::<f32>();
        assert_eq!(10, data.len());
    }

    #[test]
    fn it_fails_on_initialized_memory_read() {
        let native = Native::new();
        let context = NativeContext::new(native.devices().to_vec()).unwrap();
        let mut shared_data = SharedTensor::<f32>::new(vec![10]);

        assert_eq!(shared_data.read(&context).unwrap_err().kind(), ErrorKind::UninitializedMemory);

        assert_eq!(shared_data.read_write(&context).unwrap_err().kind(), ErrorKind::UninitializedMemory);

        shared_data.write_only(&context).unwrap();
        shared_data.drop_context(&context).unwrap();

        assert_eq!(shared_data.read(&context).unwrap_err().kind(), ErrorKind::UninitializedMemory);
    }
}

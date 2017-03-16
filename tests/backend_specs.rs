extern crate parenchyma as pa;

#[cfg(test)]
mod backend_spec {
    mod native {
        use pa::{Backend, Native};

        #[test]
        fn it_can_create_default_backend() {
            let backend: Result<Backend, _> = Backend::new::<Native>();
            assert!(backend.is_ok());
        }
    }
}
extern crate parenchyma as pa;

#[cfg(test)]
mod backend_spec {
    mod native {
        use std::rc::Rc;
        use pa::{Backend, GenericBackend, Native};

        #[test]
        fn it_can_create_default_backend() {
            let backend: Result<Backend, _> = Backend::new::<Native>();
            assert!(backend.is_ok());
        }

        #[test]
        fn it_can_use_ibackend_trait_object() {
            let backend: Rc<Backend> = Rc::new(Backend::new::<Native>().unwrap());
            use_ibackend(backend);
        }

        fn use_ibackend<B: GenericBackend>(backend: Rc<B>) {
            let backend: Rc<GenericBackend> = backend.clone();
            let _ = backend.context();
        }
    }

    // #[cfg(feature = "cuda")]
    // mod cuda {
    //     use co::*;
    //     #[test]
    //     fn it_can_create_default_backend() {
    //         assert!(Backend::new::<Cuda>().is_ok());
    //     }
    // }

    mod opencl {
        use pa::{Backend, Framework, FrameworkCtor, OpenCL};

        #[test]
        fn it_can_create_default_backend() {
            let backend: Result<Backend, _> = Backend::new::<OpenCL>();
            assert!(backend.is_ok());
        }

        #[test]
        fn it_can_manually_create_backend() {
            let framework = OpenCL::new().unwrap();
            let hardware = framework.available_hardware().to_vec();
            let backend: Backend = Backend::with(framework, hardware).unwrap();
            println!("{:?}", backend);
        }
    }
}
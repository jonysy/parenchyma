extern crate parenchyma;

#[cfg(test)]
mod framework_native_spec {
    use parenchyma::frameworks::Native;
    use parenchyma::prelude::{Framework, FrameworkCtor};

    #[test]
    fn it_works() {
        let framework: Native = Native::new().unwrap();
        assert_eq!(framework.hardware().len(), 1);
    }
}
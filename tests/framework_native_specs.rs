extern crate parenchyma as pa;

#[cfg(test)]
mod framework_native_spec {
    use pa::{Native, Framework, FrameworkCtor};

    #[test]
    fn it_works() {
        let framework = Native::new().unwrap();
        assert_eq!(framework.available_hardware().len(), 1);
    }
}
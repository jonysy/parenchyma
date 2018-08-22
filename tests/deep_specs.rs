#![feature(rustc_private)]

#[macro_use]
extern crate lazy_static;
extern crate parenchyma;
extern crate parenchyma_deep;

#[cfg(test)]
mod deep_specification_native {
    use parenchyma::frameworks::Native;
    use parenchyma::prelude::*;
    use parenchyma_deep::*;

    struct TestBackend(Backend<Package>);
    impl ::std::ops::Deref for TestBackend {
        type Target = Backend<Package>;
        fn deref(&self) -> &Self::Target { &self.0 }
    }
    unsafe impl Sync for TestBackend { }

    lazy_static! {
        static ref BACKEND: TestBackend = TestBackend(Backend::new::<Native<_>>().unwrap());
    }

    fn get_memory() -> (SharedTensor, SharedTensor) {
        let x = SharedTensor::with([1, 1, 3], &[1., 1., 2.][..]).unwrap();
        let result: SharedTensor = SharedTensor::from([1, 1, 3]);
        (x, result)
    }

    fn get_grad_memory() -> (SharedTensor, SharedTensor, SharedTensor, SharedTensor){
        let x = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let x_diff = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let result = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let result_diff = SharedTensor::from([1, 1, 3]);
        (x, x_diff, result, result_diff)
    }

    fn get_memory_softmax() -> (SharedTensor, SharedTensor) {
        let x = SharedTensor::with([1, 1, 4], vec![1.0; 4]).unwrap();
        let result: SharedTensor = SharedTensor::from([1, 1, 4]);
        (x, result)
    }

    #[test]
    fn it_computes_correct_log_softmax_on_for_f32() {
        let (mut x, mut result) = get_memory_softmax();
        BACKEND.log_softmax(&mut x, &mut result).unwrap();
        assert_eq!(&[-1.3862944, -1.3862944, -1.3862944, -1.3862944], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_log_softmax_grad_on_for_f32() {
        let (mut x, mut x_diff, _, mut result_diff) = get_grad_memory();
        BACKEND.log_softmax_grad(&mut x, &mut x_diff, &mut result_diff).unwrap();
        assert_eq!(&[-9.873127, -9.873127, -27.556225], result_diff.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_relu_on_for_f32() {
        let (mut x, mut result) = get_memory();
        BACKEND.relu(&mut x, &mut result).unwrap();
        assert_eq!(&[1., 1., 2.], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_relu_grad_on_for_f32() {
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory();
        BACKEND.relu_grad(&mut x, &mut x_diff, &mut result, &mut result_diff).unwrap();
        assert_eq!(&[1., 1., 2.], result_diff.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_sigmoid_on_for_f32() {
        let (mut x, mut result) = get_memory();
        BACKEND.sigmoid(&mut x, &mut result).unwrap();
        assert_eq!(&[0.7310585786, 0.7310586, 0.880797], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_for_f32() {
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory();
        BACKEND.sigmoid_grad(&mut x, &mut x_diff, &mut result, &mut result_diff).unwrap();
        assert_eq!(&[0., 0., -4.], result_diff.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_softmax_on_for_f32() {
        let (mut x, mut result) = get_memory_softmax();
        BACKEND.softmax(&mut x, &mut result).unwrap();
        assert_eq!(&[0.25, 0.25, 0.25, 0.25], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_softmax_grad_on_for_f32() {
        let (mut x, mut x_diff, _, mut result_diff) = get_grad_memory();
        BACKEND.softmax_grad(&mut x, &mut x_diff, &mut result_diff).unwrap();
        assert_eq!(&[-5., -5., -8.], result_diff.as_slice().unwrap());
    }
}

#[cfg(test)]
mod deep_specification_opencl {
    use parenchyma::frameworks::OpenCL;
    use parenchyma::hardware::{Hardware, HardwareKind};
    use parenchyma::prelude::*;
    use parenchyma_deep::*;

    struct TestBackend(Backend<Package>);
    impl ::std::ops::Deref for TestBackend {
        type Target = Backend<Package>;
        fn deref(&self) -> &Self::Target { &self.0 }
    }
    unsafe impl Sync for TestBackend { }

    lazy_static! {
        static ref BACKEND: TestBackend = {
            let mut backend: Backend<Package> = Backend::new::<OpenCL<_>>().unwrap();
            // required here!
            backend.select(&|hardware| hardware.kind == HardwareKind::GPU);
            TestBackend(backend)
        };
    }

    fn get_memory() -> (SharedTensor, SharedTensor) {
        let x = SharedTensor::with([1, 1, 3], &[1., 1., 2.][..]).unwrap();
        let result: SharedTensor = SharedTensor::from([1, 1, 3]);
        (x, result)
    }

    fn get_grad_memory() -> (SharedTensor, SharedTensor, SharedTensor, SharedTensor){
        let x = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let x_diff = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let result = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let result_diff = SharedTensor::from([1, 1, 3]);
        (x, x_diff, result, result_diff)
    }

    fn get_memory_softmax() -> (SharedTensor, SharedTensor) {
        let x = SharedTensor::with([1, 1, 4], vec![1.0; 4]).unwrap();
        let result: SharedTensor = SharedTensor::from([1, 1, 4]);
        (x, result)
    }

    #[test]
    fn it_computes_correct_log_softmax_on_for_f32() {
        let (mut x, mut result) = get_memory_softmax();
        BACKEND.log_softmax(&mut x, &mut result).unwrap();
        assert_eq!(&[-1.3862944, -1.3862944, -1.3862944, -1.3862944], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_relu_on_for_f32() {
        let (mut x, mut result) = get_memory();
        BACKEND.relu(&mut x, &mut result).unwrap();
        assert_eq!(&[1., 1., 2.], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_sigmoid_on_for_f32() {
        let (mut x, mut result) = get_memory();
        BACKEND.sigmoid(&mut x, &mut result).unwrap();
        assert_eq!(&[0.7310585786, 0.7310586, 0.880797], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_softmax_on_for_f32() {
        let (mut x, mut result) = get_memory_softmax();
        BACKEND.softmax(&mut x, &mut result).unwrap();
        assert_eq!(&[0.25, 0.25, 0.25, 0.25], result.as_slice().unwrap());
    }
}

#[cfg(test)]
mod deep_specification_backward_opencl {
    use parenchyma::frameworks::OpenCL;
    use parenchyma::hardware::{Hardware, HardwareKind};
    use parenchyma::prelude::*;
    use parenchyma_deep::*;

    struct TestBackend(Backend<Package>);
    impl ::std::ops::Deref for TestBackend {
        type Target = Backend<Package>;
        fn deref(&self) -> &Self::Target { &self.0 }
    }
    unsafe impl Sync for TestBackend { }

    lazy_static! {
        static ref BACKEND: TestBackend = {
            let mut backend: Backend<Package> = Backend::new::<OpenCL<_>>().unwrap();
            // required here!
            backend.select(&|hardware| hardware.kind == HardwareKind::GPU);
            TestBackend(backend)
        };
    }

    fn get_memory() -> (SharedTensor, SharedTensor) {
        let x = SharedTensor::with([1, 1, 3], &[1., 1., 2.][..]).unwrap();
        let result: SharedTensor = SharedTensor::from([1, 1, 3]);
        (x, result)
    }

    fn get_grad_memory() -> (SharedTensor, SharedTensor, SharedTensor, SharedTensor){
        let x = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let x_diff = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let result = SharedTensor::with([1, 1, 3], &[1.0, 1.0, 2.0][..]).unwrap();
        let result_diff = SharedTensor::from([1, 1, 3]);
        (x, x_diff, result, result_diff)
    }

    #[test]
    fn it_computes_correct_log_softmax_grad_on_for_f32() {
        let (mut x, mut x_diff, _, mut result_diff) = get_grad_memory();
        BACKEND.log_softmax_grad(&mut x, &mut x_diff, &mut result_diff).unwrap();
        assert_eq!(&[-9.873127, -9.873127, -27.556223], result_diff.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_for_f32() {
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory();
        BACKEND.sigmoid_grad(&mut x, &mut x_diff, &mut result, &mut result_diff).unwrap();
        assert_eq!(&[0., 0., -4.], result_diff.as_slice().unwrap());
    }
}
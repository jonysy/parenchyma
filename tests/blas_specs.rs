#![feature(rustc_private)]

#[macro_use]
extern crate lazy_static;
#[macro_use(array)]
extern crate parenchyma;
extern crate parenchyma_blas;

#[cfg(test)]
mod blas_specification_native {
    use parenchyma::frameworks::Native;
    use parenchyma::prelude::*;
    use parenchyma_blas::*;

    struct TestBackend(Backend<Package>);

    impl ::std::ops::Deref for TestBackend {
        type Target = Backend<Package>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    unsafe impl Sync for TestBackend { }

    lazy_static! {
        static ref BACKEND: TestBackend = TestBackend(Backend::new::<Native>().unwrap());
    }

    #[test]
    fn it_computes_correct_asum_on_native_for_f32() {
        let ref x = array![1., -2., 3.].into();
        let ref mut result = SharedTensor::scalar(0.0).unwrap();
        BACKEND.asum(x, result).unwrap();
        assert_eq!(&[6.], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_axpy_on_native_for_f32() {
        let ref a = SharedTensor::scalar(2.0).unwrap();
        let ref x = array![1., 2., 3.].into();
        let ref mut y = array![1., 2., 3.].into();
        BACKEND.axpy(a, x, y).unwrap();
        assert_eq!(&[3., 6., 9.], y.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_copy_on_native_for_f32() {
        let ref mut x = array![1., 2., 3.].into();
        let ref mut y = SharedTensor::from([3]);
        BACKEND.copy(x, y).unwrap();
        assert_eq!(&[1., 2., 3.], y.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_dot_on_native_for_f32() {
        let ref x = array![1., 2., 3.].into();
        let ref y = array![1., 2., 3.].into();
        let ref mut result = SharedTensor::from([]);
        BACKEND.dot(x, y, result).unwrap();
        assert_eq!(&[14.], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_nrm2_on_native_for_f32() {
        let ref x = array![1., 2., 2.].into();
        let ref mut result = SharedTensor::from([]);
        BACKEND.nrm2(x, result).unwrap();
        assert_eq!(&[3.], result.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_scal_on_native_for_f32() {
        let ref a = array![2.].into();
        let ref mut x = array![1., 2., 3.].into();
        BACKEND.scal(a, x).unwrap();
        assert_eq!(&[2., 4., 6.], x.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_swap_on_native_for_f32() {
        let ref mut x = array![1., 2., 3.].into();
        let ref mut y = array![3., 2., 1.].into();
        BACKEND.swap(x, y).unwrap();
        assert_eq!(&[3., 2., 1.], x.as_slice().unwrap());
        assert_eq!(&[1., 2., 3.], y.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_gemm_on_native_for_f32() {

        let ref alpha = array![1.0].into();
        let ref amat = 
            array![
                [2.0, 5.0], 
                [2.0, 5.0], 
                [2.0, 5.0]
            ].into();

        let ref beta = array![0.0].into();
        let ref bmat =
            array![
                [4.0, 1.0, 1.0],
                [4.0, 1.0, 1.0]
            ].into();

        let ref mut cmat = SharedTensor::from([3, 3]);
        let transposition = Transposition::NoTranspose;

        BACKEND.gemm(alpha, amat, transposition, beta, bmat, transposition, cmat).unwrap();

        assert_eq!(&[28., 7., 7., 28., 7., 7., 28., 7., 7.], cmat.as_slice().unwrap());
    }

    #[test]
    fn it_computes_correct_transpose_gemm_on_native_for_f32() {
        
        let ref alpha = array![1.0].into();
        let ref amat =
            array![
                [2.0, 5.0], 
                [2.0, 5.0], 
                [2.0, 5.0]
            ].into();

        let ref beta = array![0.0].into();
        let ref bmat =
            array![
                [4.0, 1.0, 1.0],
                [4.0, 1.0, 1.0]
            ].into();

        let ref mut cmat = SharedTensor::from([2, 2]);
        let transposition = Transposition::Transpose;

        BACKEND.gemm(alpha, amat, transposition, beta, bmat, transposition, cmat).unwrap();

        assert_eq!(&[12., 12., 30., 30.], cmat.as_slice().unwrap());
    }
}
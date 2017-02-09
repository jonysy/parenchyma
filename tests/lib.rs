#![cfg(test)]

extern crate parenchyma;

mod backend_spec {
    use parenchyma::{Backend, Native};

    #[test]
    fn it_can_create_default_backend() {

        assert!(Backend::<Native>::default().is_ok());
    }
}

mod framework_spec {
    use parenchyma::{Framework, Native};

    #[test]
    fn it_can_init_native_framework() {
        let framework = Native::new().unwrap();
        assert_eq!(framework.available_devices.len(), 1);
    }
}

// mod shared_memory_spec {
//     use parenchyma::{Context, Framework, SharedTensor};
//     use parenchyma::error::ErrorKind;
//     use parenchyma_native::{Native, NativeContext};

//     #[test]
//     fn it_creates_new_shared_memory_for_native() {
//         let native = Native::new();
//         let context = NativeContext::new(native.devices().to_vec()).unwrap();
//         let mut shared_data = SharedTensor::<f32>::new(vec![10]);
//         let data = shared_data.write_only(&context).unwrap().as_slice::<f32>();
//         assert_eq!(10, data.len());
//     }

//     #[test]
//     fn it_fails_on_initialized_memory_read() {
//         let native = Native::new();
//         let context = NativeContext::new(native.devices().to_vec()).unwrap();
//         let mut shared_data = SharedTensor::<f32>::new(vec![10]);

//         assert_eq!(shared_data.read(&context).unwrap_err().kind(), ErrorKind::UninitializedMemory);

//         assert_eq!(shared_data.read_write(&context).unwrap_err().kind(), ErrorKind::UninitializedMemory);

//         shared_data.write_only(&context).unwrap();
//         shared_data.drop_context(&context).unwrap();

//         assert_eq!(shared_data.read(&context).unwrap_err().kind(), ErrorKind::UninitializedMemory);
//     }
// }

// #[cfg(test)]
// mod tensor_spec {
//     use parenchyma::{SharedTensor, Tensor};

//     #[test]
//     fn it_returns_correct_tensor_desc_stride() {

//         let tensor_desc_r0 = Tensor::from(vec![]);
//         let tensor_desc_r1 = Tensor::from(vec![5]);
//         let tensor_desc_r2 = Tensor::from(vec![2, 4]);
//         let tensor_desc_r3 = Tensor::from(vec![2, 2, 4]);
//         let tensor_desc_r4 = Tensor::from(vec![2, 2, 4, 4]);

//         assert!(vec![0; 0] == tensor_desc_r0.default_stride());
//         assert_eq!(vec![1], tensor_desc_r1.default_stride());
//         assert_eq!(vec![4, 1], tensor_desc_r2.default_stride());
//         assert_eq!(vec![8, 4, 1], tensor_desc_r3.default_stride());
//         assert_eq!(vec![32, 16, 4, 1], tensor_desc_r4.default_stride());
//     }

//     #[test]
//     fn it_returns_correct_size_for_rank_0() {
//         // In order for memory to be correctly allocated, the size should never return less than 1.
//         let tensor_desc_r0 = Tensor::from(vec![]);
//         assert_eq!(1, tensor_desc_r0.ncomponents());
//     }

//     #[test]
//     fn it_resizes_tensor() {
//         let mut shared_tensor = SharedTensor::<f32>::new(vec![10, 20, 30]);
//         assert_eq!(shared_tensor.shape(), &[10, 20, 30]);

//         shared_tensor.replace(vec![2, 3, 4, 5]);
//         assert_eq!(shared_tensor.shape(), &[2, 3, 4, 5]);
//     }

//     #[test]
//     fn it_reshapes_correctly() {
//         let mut shared_data = SharedTensor::<f32>::new(vec![10]);
//         assert!(shared_data.reshape(vec![5, 2]).is_ok());
//     }

//     #[test]
//     fn it_returns_err_for_invalid_size_reshape() {
//         let mut shared_data = SharedTensor::<f32>::new(vec![10]);
//         assert!(shared_data.reshape(vec![10, 2]).is_err());
//     }
// }

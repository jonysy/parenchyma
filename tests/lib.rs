extern crate parenchyma;

#[cfg(test)]
mod tensor_spec {
	use parenchyma::{SharedTensor, Tensor};

	#[test]
	fn it_returns_correct_tensor_desc_stride() {

		let tensor_desc_r0 = Tensor::from(vec![]);
		let tensor_desc_r1 = Tensor::from(vec![5]);
		let tensor_desc_r2 = Tensor::from(vec![2, 4]);
		let tensor_desc_r3 = Tensor::from(vec![2, 2, 4]);
		let tensor_desc_r4 = Tensor::from(vec![2, 2, 4, 4]);

		assert!(vec![0; 0] == tensor_desc_r0.default_stride());
		assert_eq!(vec![1], tensor_desc_r1.default_stride());
		assert_eq!(vec![4, 1], tensor_desc_r2.default_stride());
		assert_eq!(vec![8, 4, 1], tensor_desc_r3.default_stride());
		assert_eq!(vec![32, 16, 4, 1], tensor_desc_r4.default_stride());
	}

	#[test]
	fn it_returns_correct_size_for_rank_0() {
		// In order for memory to be correctly allocated, the size should never return less than 1.
		let tensor_desc_r0 = Tensor::from(vec![]);
		assert_eq!(1, tensor_desc_r0.ncomponents());
	}

	#[test]
	fn it_resizes_tensor() {
		let mut shared_tensor = SharedTensor::<f32>::new(vec![10, 20, 30]);
		assert_eq!(shared_tensor.shape(), &[10, 20, 30]);

		shared_tensor.replace(vec![2, 3, 4, 5]);
		assert_eq!(shared_tensor.shape(), &[2, 3, 4, 5]);
	}

	#[test]
	fn it_reshapes_correctly() {
		let mut shared_data = SharedTensor::<f32>::new(vec![10]);
		assert!(shared_data.reshape(vec![5, 2]).is_ok());
	}

	#[test]
	fn it_returns_err_for_invalid_size_reshape() {
		let mut shared_data = SharedTensor::<f32>::new(vec![10]);
		assert!(shared_data.reshape(vec![10, 2]).is_err());
	}
}
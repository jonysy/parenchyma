#![cfg(test)]

extern crate parenchyma;
extern crate parenchyma_cuda as cuda;

mod shared_memory_spec {
    use cuda::Cuda;
    use parenchyma::{Backend, Native, NativeMemory, SharedTensor};

    fn write_to_memory<T: Copy>(mem: &mut NativeMemory, data: &[T]) {

        let buffer = mem.as_mut_slice::<T>();

        for (i, datum) in data.iter().enumerate() {

            buffer[i] = *datum;
        }
    }

    #[test]
    fn it_creates_new_shared_memory_for_cuda() {

        let backend: Backend<Cuda> = Backend::default().unwrap();
        let shape = vec![10];
        let mut shared_data = SharedTensor::<f32>::from(shape);
        assert!(shared_data.write_only(backend.context()).is_ok())
    }

    #[test]
    fn it_syncs_from_native_to_cuda_and_back() {
        let shape = vec![3];

        let cuda: Backend<Cuda> = Backend::default().unwrap();

        let native: Backend<Native> = Backend::default().unwrap();

        let mut mem: SharedTensor<f64> = SharedTensor::from(shape);

        write_to_memory(
            mem.write_only(native.context()).unwrap(), 
            &[1.0, 2.0, 123.456]
        );

        assert!(mem.read(cuda.context()).is_ok());

        // It has not successfully synced to the device.
        // Not the other way around.
        assert!(mem.drop_context(native.context()).is_ok());

        assert_eq!(mem.read(native.context()).unwrap().as_slice::<f64>(), [1.0, 2.0, 123.456]);
    }
}
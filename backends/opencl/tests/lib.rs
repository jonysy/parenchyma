#![cfg(test)]

extern crate parenchyma;
extern crate parenchyma_opencl as opencl;

mod shared_memory_spec {
    use opencl::{OpenCL};
    use parenchyma::{Backend, Framework, Native, NativeMemory, Tensor};

    fn write_to_memory<T: Copy>(mem: &mut NativeMemory, data: &[T]) {

        let buffer = mem.as_mut_slice::<T>();

        for (i, datum) in data.iter().enumerate() {

            buffer[i] = *datum;
        }
    }

    #[test]
    fn it_creates_new_shared_memory_for_opencl() {

        let backend: Backend<OpenCL> = Backend::default().unwrap();
        let shape = vec![10];
        let mut shared_data = Tensor::<f32>::from(shape);
        assert!(shared_data.write_only(backend.context()).is_ok())
    }

    #[test]
    fn it_syncs_from_native_to_opencl_and_back0() {
        let shape = vec![3];

        let framework = OpenCL::new().unwrap();
        let selection = framework.available_platforms[0].available_devices[0].clone();
        let cl: Backend<OpenCL> = Backend::new(framework, selection).unwrap();

        let native: Backend<Native> = Backend::default().unwrap();

        let mut mem: Tensor<f64> = Tensor::from(shape);

        write_to_memory(
            mem.write_only(native.context()).unwrap(), 
            &[1.0, 2.0, 123.456]
        );

        assert!(mem.read(cl.context()).is_ok());

        // It has not successfully synced to the device.
        // Not the other way around.
        assert!(mem.drop_context(native.context()).is_ok());

        assert_eq!(mem.read(native.context()).unwrap().as_slice::<f64>(), [1.0, 2.0, 123.456]);
    }

    #[test]
    fn it_syncs_from_native_to_opencl_and_back1() {
        let shape = vec![3];

        let framework = OpenCL::new().unwrap();
        let selection = framework.available_platforms[0].available_devices[1].clone();
        let cl: Backend<OpenCL> = Backend::new(framework, selection).unwrap();

        let native: Backend<Native> = Backend::default().unwrap();

        let mut mem: Tensor<f64> = Tensor::from(shape);

        write_to_memory(
            mem.write_only(native.context()).unwrap(), 
            &[1.0, 2.0, 123.456]
        );

        assert!(mem.read(cl.context()).is_ok());

        // It has not successfully synced to the device.
        // Not the other way around.
        assert!(mem.drop_context(native.context()).is_ok());

        assert_eq!(mem.read(native.context()).unwrap().as_slice::<f64>(), [1.0, 2.0, 123.456]);
    }
}
// extern crate parenchyma as pa;

// #[cfg(test)]
// mod shared_memory_spec {
//     use pa::{Backend, ErrorKind, Memory, Native, OpenCL, SharedTensor};

//     pub fn write(memory: &mut Memory<f32>, data: &[f32]) {
//         let ndarray = unsafe { memory.as_mut_native_unchecked() };
//         let buf = ndarray.as_slice_memory_order_mut().unwrap();

//         for (index, datum) in data.iter().enumerate() {
//             buf[index] = *datum;
//         }
//     }

//     #[test]
//     fn it_creates_new_shared_memory_for_native() {
//         let ref host: Backend = Backend::new::<Native>().unwrap();
//         let mut shared_data = SharedTensor::<f32>::new(10);
//         let tensor = shared_data.write(host).unwrap();
//         assert_eq!(tensor.as_native().unwrap().len(), 10);
//     }

//     #[test]
//     //#[cfg(feature = "opencl")]
//     fn it_creates_new_shared_memory_for_opencl() {
//         let ref backend: Backend = Backend::new::<OpenCL>().unwrap();
//         let mut shared_data: SharedTensor = SharedTensor::new(10);
//         assert!(shared_data.write(backend).is_ok());
//     }

//     #[test]
//     fn it_fails_on_initialized_memory_read() {
//         let ref host: Backend = Backend::new::<Native>().unwrap();
//         let mut shared_data = SharedTensor::<f32>::new(10);
//         assert_eq!(shared_data.read(host).unwrap_err().kind(), ErrorKind::UninitializedMemory);
//         assert_eq!(shared_data.read_write(host).unwrap_err().kind(), ErrorKind::UninitializedMemory);

//         // initialize memory
//         let _ = shared_data.write(host).unwrap();
//         let _ = shared_data.dealloc(host).unwrap();

//         assert_eq!(shared_data.read(host).unwrap_err().kind(), ErrorKind::UninitializedMemory);
//     }

//     #[test]
//     //#[cfg(feature = "opencl")]
//     fn it_syncs_from_native_to_opencl_and_back() {
//         let ref host: Backend = Backend::new::<Native>().unwrap();
//         let ref backend: Backend = Backend::new::<OpenCL>().unwrap();

//         let mut sh = SharedTensor::<f32>::new(3);
//         write(sh.write(host).unwrap(), &[1.0f32, 2.0, 123.456]);
//         let _ = sh.read(backend).unwrap();

//         // It has not successfully synced to the device.
//         // Not the other way around.

//         //let _ = sh.dealloc(host).unwrap();// TODO ?
//         let _ = sh.dealloc(backend).unwrap();

//         assert_eq!(
//             sh.read(host).unwrap().as_native().unwrap().as_slice_memory_order().unwrap(), 
//             [1.0, 2.0, 123.456]
//         );
//     }

//     #[test]
//     fn it_reshapes_correctly() {
//         let mut shared_data = SharedTensor::<f32>::new(10);
//         assert!(shared_data.reshape([5, 2]).is_ok());
//     }

//     #[test]
//     fn it_returns_err_for_invalid_size_reshape() {
//         let mut shared_data = SharedTensor::<f32>::new(10);
//         assert!(shared_data.reshape([10, 2]).is_err());
//     }
// }
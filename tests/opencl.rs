// #![cfg(test)]

// extern crate parenchyma;
// extern crate parenchyma_opencl as opencl;

// mod shared_memory_spec {
//     use opencl::OpenCL;
//     use parenchyma::{Backend, DeviceKind, Framework, Native, NativeMemory, SharedTensor};

//     fn write_to_memory<T: Copy>(mem: &mut NativeMemory, data: &[T]) {

//         let buffer = mem.as_mut_slice::<T>();

//         for (i, datum) in data.iter().enumerate() {

//             buffer[i] = *datum;
//         }
//     }

//     #[test]
//     fn it_creates_new_shared_memory_for_opencl() {

//         let backend: Backend<OpenCL> = Backend::default().unwrap();
//         let shape = vec![10];
//         let mut shared_data = SharedTensor::<f32>::from(shape);
//         assert!(shared_data.write_only(&backend.devices()[0]).is_ok())
//     }

//     #[test]
//     fn it_syncs_from_native_to_opencl_and_back_cpu() {
//         let shape = vec![3];

//         let framework = OpenCL::new().unwrap();
//         let selection = framework.available_platforms[0].available_devices.clone();
//         let cl: Backend<OpenCL> = Backend::new(framework, selection).unwrap();
//         let cl_cpu = cl.devices().iter().filter(|d| *d.kind() == DeviceKind::Cpu).nth(0).unwrap();

//         let native: Backend<Native> = Backend::default().unwrap();

//         let mut mem: SharedTensor<f64> = SharedTensor::from(shape);

//         write_to_memory(
//             mem.write_only(&native.devices()[0]).unwrap(), 
//             &[1.0, 2.0, 123.456]
//         );

//         assert!(mem.read(cl_cpu).is_ok());

//         // It has not successfully synced to the device.
//         // Not the other way around.
//         assert!(mem.drop_device(&native.devices()[0]).is_ok());

//         assert_eq!(mem.read(&native.devices()[0]).unwrap().as_slice::<f64>(), [1.0, 2.0, 123.456]);
//     }

//     #[test]
//     fn it_syncs_from_native_to_opencl_and_back_gpu() {
//         let shape = vec![3];

//         let framework = OpenCL::new().unwrap();
//         let selection = framework.available_platforms[0].available_devices.clone();
//         let cl: Backend<OpenCL> = Backend::new(framework, selection).unwrap();
//         let cl_gpu = cl.devices().iter().filter(|d| *d.kind() == DeviceKind::Gpu).nth(0).unwrap();

//         let native: Backend<Native> = Backend::default().unwrap();

//         let mut mem: SharedTensor<f64> = SharedTensor::from(shape);

//         write_to_memory(
//             mem.write_only(&native.devices()[0]).unwrap(), 
//             &[1.0, 2.0, 123.456]
//         );

//         assert!(mem.read(cl_gpu).is_ok());

//         // It has not successfully synced to the device.
//         // Not the other way around.
//         assert!(mem.drop_device(&native.devices()[0]).is_ok());

//         assert_eq!(mem.read(&native.devices()[0]).unwrap().as_slice::<f64>(), [1.0, 2.0, 123.456]);
//     }
// }
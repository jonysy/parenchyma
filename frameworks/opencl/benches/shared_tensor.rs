#![feature(test)]

extern crate parenchyma;
extern crate parenchyma_opencl;
extern crate test;

use parenchyma::{Backend, Device, Framework, Native, SharedTensor};
use parenchyma::DeviceKind::{Cpu, Gpu};
use parenchyma_opencl::{OpenCL, OpenCLDevice};
use test::Bencher;

fn native_backend() -> Backend<Native> {
    Backend::default().unwrap()
}

fn opencl_backend() -> Backend<OpenCL> {
    Backend::default().unwrap()
}

fn sync_back_and_forth<A, B>(b: &mut Bencher, backend1: Backend<A>, backend2: Backend<B>, s: usize) 
    where A: Framework, B: Framework,
{
    let ref dev1 = backend1.devices()[0];
    let ref dev2 = backend2.devices()[0];

    let mem = &mut SharedTensor::<u8>::from(vec![s]);

    // initialize and warm-up
    let _ = mem.write_only(dev2).unwrap();
    let _ = mem.read_write(dev1).unwrap();
    let _ = mem.read_write(dev2).unwrap();

    b.bytes = s as u64 * 2; // we do two transfers per iteration

    b.iter(|| {
        let _ = mem.read_write(dev1).unwrap();
        let _ = mem.read_write(dev2).unwrap();
    });
}

fn unidirectional_sync<A, B>(b: &mut Bencher, src: Backend<A>, dst: Backend<B>, size: usize)
    where A: Framework, B: Framework,
{
    let ref src_dev = src.devices()[0];
    let ref dst_dev = dst.devices()[0];

    let mem = &mut SharedTensor::<u8>::from(vec![size]);

    // initialize and warm-up
    let _ = mem.write_only(src_dev).unwrap();
    let _ = mem.read(dst_dev).unwrap();

    b.bytes = size as u64;

    b.iter(|| {
        let _ = mem.write_only(src_dev).unwrap();
        let _ = mem.read(dst_dev).unwrap();
    });
}

#[inline(never)]
fn bench_256_alloc_1mb_opencl_profile(b: &mut Bencher, device: &OpenCLDevice, size: usize) {
    b.iter(|| 
        for _ in 0..256 {
            let _ = device.allocate_memory(size).unwrap(); });
}

// #[bench]
// fn bench_256_alloc_1mb_opencl_cpu(b: &mut Bencher) {
//     let opencl_backend = opencl_backend();
//     let cpu = opencl_backend.devices().iter().filter(|d| *d.kind() == Cpu).nth(0).unwrap();

//     bench_256_alloc_1mb_opencl_profile(b, cpu, 1_048_576);
// }

// #[bench]
// fn bench_256_alloc_1mb_opencl_gpu(b: &mut Bencher) {
//     let opencl_backend = opencl_backend();
//     let gpu = opencl_backend.devices().iter().filter(|d| *d.kind() == Gpu).nth(0).unwrap();

//     bench_256_alloc_1mb_opencl_profile(b, gpu, 1_048_576);
// }

#[bench]
fn bench_256_alloc_1mb_opencl(b: &mut Bencher) {
    let opencl_backend = opencl_backend();
    let ref d = opencl_backend.devices()[0];

    bench_256_alloc_1mb_opencl_profile(b, d, 1_048_576);
}

#[bench]
fn bench_sync_1kb_native_opencl_back_and_forth(b: &mut Bencher) {
    sync_back_and_forth(b, opencl_backend(), native_backend(), 1024);
}

#[bench]
fn bench_sync_1kb_native_to_opencl(b: &mut Bencher) {
    unidirectional_sync(b, native_backend(), opencl_backend(), 1024);
}

#[bench]
fn bench_sync_1kb_opencl_to_native(b: &mut Bencher) {
    unidirectional_sync(b, opencl_backend(), native_backend(), 1024);
}

#[bench]
fn bench_sync_1mb_native_opencl_back_and_forth(b: &mut Bencher) {
    sync_back_and_forth(b, opencl_backend(), native_backend(), 1_048_576);
}

#[bench]
fn bench_sync_1mb_native_to_opencl(b: &mut Bencher) {
    unidirectional_sync(b, native_backend(), opencl_backend(), 1_048_576);
}

#[bench]
fn bench_sync_1mb_opencl_to_native(b: &mut Bencher) {
    unidirectional_sync(b, opencl_backend(), native_backend(), 1_048_576);
}

#[bench]
fn bench_sync_128mb_native_opencl_back_and_forth(b: &mut Bencher) {
    sync_back_and_forth(b, opencl_backend(), native_backend(), 128 * 1_048_576);
}

#[bench]
fn bench_sync_128mb_native_to_opencl(b: &mut Bencher) {
    unidirectional_sync(b, native_backend(), opencl_backend(), 128 * 1_048_576);
}

#[bench]
fn bench_sync_128mb_opencl_to_native(b: &mut Bencher) {
    unidirectional_sync(b, opencl_backend(), native_backend(), 128 * 1_048_576);
}

// fn bench_shared_tensor_access_time_first_(b: &mut Bencher, device: &OpenCLDevice) {

//     let native_backend = native_backend();
//     let ref native_cpu = native_backend.devices()[0];

//     let mut x = SharedTensor::<f32>::from(vec![128]);
//     x.write_only(native_cpu).unwrap();
//     x.write_only(device).unwrap();
//     x.read(native_cpu).unwrap();

//     b.iter(|| {
//         let _ = x.read(native_cpu).unwrap();
//     })
// }

// #[bench]
// fn bench_shared_tensor_access_time_first_cpu(b: &mut Bencher) {
//     let opencl_backend = opencl_backend();
//     let opencl_cpu = opencl_backend.devices().iter().filter(|d| *d.kind() == Cpu).nth(0).unwrap();

//     bench_shared_tensor_access_time_first_(b, opencl_cpu);
// }

// #[bench]
// fn bench_shared_tensor_access_time_first_gpu(b: &mut Bencher) {
//     let opencl_backend = opencl_backend();
//     let opencl_gpu = opencl_backend.devices().iter().filter(|d| *d.kind() == Gpu).nth(0).unwrap();

//     bench_shared_tensor_access_time_first_(b, opencl_gpu);
// }
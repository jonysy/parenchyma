#![allow(warnings)]
#![feature(alloc, box_heap, box_syntax, collection_placement, heap_api, placement_in_syntax)]

extern crate alloc;
extern crate futures;
extern crate parenchyma;

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use alloc::raw_vec::RawVec;
use futures::{Async, Future};
use parenchyma::prelude::*;
use std::{mem, slice};

fn main() {
    // let framework = Native;
    // let selection = framework.hardware().to_vec();
    // let ref host: Backend = Backend::with(framework, selection);

    let framework = OpenCL::default();
    let selection = framework.hardware().to_vec();
    let ref backend: Backend = Backend::with(framework, selection);
    let ref x = SharedTensor::with(backend, [1, 3], vec![1.2, 2.3, 3.4]).unwrap();



    // io::view::<f32>(backend, x)?
    // io::write::<f32>(backend, x, &[1.2, 3.4, 6.2][..], 5..)?;
    // io::write_offset::<f32>(backend, x, &[2.3, 6.2, 1.3][..], 2..3)?;

    //println!("{:?}", io::view::<f32>(backend, x));

    // let tensor = x.read(backend).unwrap();
    // println!("I'm hip1! {:#?}", tensor);

    // unsafe {
    //     println!("{:?}", 10000.2333f32);
    //     let n = &10000.2333f32 as *const f32 as *const u8;
    //     println!("{}", *n);
    //     let n = n as *const f32;
    //     println!("{:?}", *n);
    // }



    //chyma::view::<f32>(x, backend)?

    //let t = tensor.downcast_ref::<parenchyma::opencl::MemoryLock>().unwrap();

    // println!("c");
    // let view = t.view::<f32>();
    // match view.wait() {
    //     Ok(slice) => println!("ok!"),
    //     Err(e) => panic!("{}", e),
    // }

    // let mut slice = t.view::<f32>(); // temporary view of the mapped memory

    // while let Ok(async) = slice.poll() {
    //     if async != Async::NotReady {
    //         break;
    //     }
    // }

    // println!("hi!: {:#?}", slice.wait().unwrap());

    // println!("Created the future");

    // let slice = chyma::view::<OpenCL>(backend, x).wait();

    // println!("{:?}", slice);

    // let mut x = SharedTensor::alloc(host, shape).map(|p| in p { [1.3, 2.4, 3.1] });
    // let mut x = SharedTensor::alloc(host, shape).map(|p| p <- [1.3, 2.4, 3.1]);
    // in x { [1.3, 3.6, 2.3] };

    // println!("{:?}", x.read(backend).downcast_ref::<OpenCL>().read());

    // chyma::write_to(x, &[1.3, 3.6, 2.3][..])?;
    // println!("{:#?}", chyma::read::<OpenCL>(backend, x)?.as_slice()); // `data=[1.3, 3.6, 2.3] shape=[ 1, 3]`

}
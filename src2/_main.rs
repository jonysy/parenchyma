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

const CAPACITY: usize = 10;

mod chyma {
    use super::RangeArgument;

    pub fn write<T, R>(x: &mut [T], x_range: R, data: &[T]) where T: Copy, R: RangeArgument {
        let length = x.len();
        let start = x_range.start().unwrap_or(0);
        let end = x_range.end().unwrap_or(length);

        assert!(end >= start);

        let skip = start;
        let take = end - start;

        assert_eq!(take, data.len());

        for (x_datum, &datum) in x.iter_mut().skip(skip).take(take).zip(data) {
            *x_datum = datum;
        }
    }
}

// fn main() {
//     // let mut a = [1.3, 4.5, 6.7, 3.4];
//     // let b     = [1.3, 4.5, 6.6, 3.4];

//     // chyma::io::write(x, .., &b)?;
//     // chyma::io::write_on(backend, x, .., &b)?;

//     // println!("{:?}", a);
// }






pub trait RangeArgument {

    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl RangeArgument for Range<usize> {

    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl RangeArgument for RangeFrom<usize> {

    fn start(&self) -> Option<usize> {
        Some(self.start)
    }
}

impl RangeArgument for RangeTo<usize> {

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl RangeArgument for RangeFull { }

// fn main() {
//     let mut vec: Vec<f32> = vec![1.2, 3.4, 5.6];
//     let length = vec.len();
//     let length_bytes = length * mem::size_of::<f32>();
//     /// > `vec![x; n]`, `vec![a, b, c, d]`, and `Vec::with_capacity(n)`, will all produce a `Vec` 
//     /// > with exactly the requested capacity. If `len()==capacity()`, (as is the case for 
//     /// > the `vec!` macro), then a `Vec<T>` can be converted to and from a `Box<[T]>` without 
//     /// > reallocating or moving the elements.
//     let boxed_slice: Box<[f32]> = vec.into_boxed_slice();

// }

// fn main() {
//     type T = f32;

//     let capacity = 10;
//     let msize = capacity * mem::size_of::<T>();
//     let raw = RawVec::with_capacity(msize);
//     let b: Box<[u8]> = unsafe { raw.into_box() };
//     //println!("{:?}", mem::size_of::<u8>());
//     let nbytes = b.len();
//     let raw: *mut [u8] = Box::into_raw(b);
//     let pointer: *mut T = raw as *mut T;
//     println!("{:?}", pointer);
//     let integer = pointer as usize;
//     println!("{:?}", integer);
//     let pointer = integer as *mut T;
//     println!("{:?}", pointer);

//     // let slice = unsafe { slice::from_raw_parts_mut(pointer, nbytes / mem::size_of::<T>()) };

//     // println!("{}", slice.len());
//     // println!("{:?}", slice);
// }

// pub fn write_to_memory_offset<T: NumCast + ::std::marker::Copy>(mem: &mut MemoryType, data: &[T], offset: usize) {
//     match mem {
//         &mut MemoryType::Native(ref mut mem) => {
//             let mut mem_buffer = mem.as_mut_slice::<f32>();
//             for (index, datum) in data.iter().enumerate() {
//                 // mem_buffer[index + offset] = *datum;
//                 mem_buffer[index + offset] = cast(*datum).unwrap();
//             }
//         },
//         #[cfg(any(feature = "opencl", feature = "cuda"))]
//         _ => {}
//     }
// }

fn main() {
    let framework = Native;
    let selection = framework.hardware().to_vec();
    let ref host: Backend = Backend::with(framework, selection);

    // // let framework = OpenCL::default();
    // // let selection = framework.hardware().to_vec();
    // // let ref backend: Backend = Backend::with(framework, selection);
    // let ref x = SharedTensor::with(backend, [1, 3], vec![1.2, 2.3, 3.4]).unwrap();



    // io::view::<f32>(backend, x)?
    // io::write::<f32>(backend, x, &[1.2, 3.4, 6.2][..], 5..)?;
    // io::write_offset::<f32>(backend, x, &[2.3, 6.2, 1.3][..], 2..3)?;

    //println!("{:?}", io::view::<f32>(backend, x));

    // let tensor = x.read(backend).unwrap();
    // println!("{:#?}", tensor.bytes());

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
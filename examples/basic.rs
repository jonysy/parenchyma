extern crate parenchyma as pa;

use pa::{Backend, Native, SharedTensor};

fn main() {
    let ref native: Backend = Backend::new::<Native>().unwrap();

    let data: Vec<f32> = vec![3.5, 12.4, 0.5, 6.5];
    let shape = [2, 2];

    let ref x = SharedTensor::with(native, shape, data).unwrap();

    println!("{}", x.read(native).unwrap().as_native().unwrap());
}
#![feature(conservative_impl_trait)]

extern crate parenchyma;

use parenchyma::{Backend, Shape, SharedTensor, Tensor};

fn main() {
    let selection = 0;
    let framework = OpenCl::new()?;
    let backend = Bakcend::new(framework, selection)?;

    let x = SharedTensor::new(&backend, [1usize], &[1.0]);
    let result = SharedTensor::new(&backend, [1usize], &[0.0]);

    ops::sigmoid(&x, &result)?;

    let tensor = result.view(&backend)?;

    println!("{:?}", tensor);
}
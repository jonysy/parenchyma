extern crate parenchyma;

use parenchyma::{Backend, SharedTensor};

fn main() {
    let backend = Backend::new().expect("failed to construct backend");
    let x = SharedTensor::new(&backend, [1], &mut [10.0]).expect("failed to alloc memory");
    let mut result = SharedTensor::new(&backend, [1], &mut [0.0]).expect("failed to alloc memory");

    parenchyma::ops::sigmoid(&backend, &x, &result).expect("failed to compute");

    let tensor = result.view(&backend).expect("failed to construct tensor");
    let output = tensor.buffer()[0];

    // TODO 
    // http://floating-point-gui.de/errors/comparison/
    // https://randomascii.wordpress.com/2014/01/27/theres-only-four-billion-floatsso-test-them-all/

    assert!((output - 0.99995460213).abs() < 0.000000000001);

    println!("{}", output);
}
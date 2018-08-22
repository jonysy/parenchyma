//! A Parenchyma package that bundles the BLAS and Deep NN packages together to make one convenient
//! ML package.
//!
//! # Example Usage
//!
//! ```ignore
//! extern crate parenchyma;
//! extern crate parenchyma_ml;
//! 
//! #[macro_use]
//! use parenchyma::prelude::*;
//! use extension_package::package::Package as MachLrnPackage;
//! 
//! // Initialize an OpenCL or CUDA backend packaged with the NN extension.
//! let backend = BackendConfig::<MachLrnPackage>::new::<OpenCL>()?;
//! 
//! // Initialize two tensors.
//! let ref x: SharedTensor = array![3.5, 12.4, 0.5, 6.5].into();
//! let ref mut result: SharedTensor = data.shape().into();
//! 
//! // Run the sigmoid operation, provided by the NN extension, on your OpenCL/CUDA enabled 
//! // GPU (or CPU, which is possible through OpenCL)
//! backend.sigmoid(x, result)?;
//! 
//! // Print the result: `[0.97068775, 0.9999959, 0.62245935, 0.9984988] shape=[4], strides=[1]`
//! println!("{:?}", result);
//! ```

extern crate parenchyma;
extern crate parenchyma_blas;
extern crate parenchyma_deep;

pub use self::extension_package::{Dependencies, Extension, Package};

mod extension_package;
mod frameworks;
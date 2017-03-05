//! CUDA backend support.

pub mod hl;
pub mod sh;

/// Provides the CUDA framework.
#[derive(Debug)]
pub struct Cuda {
    // /// A list of available devices.
    // pub available_devices: Vec<CudaDevice>,
}
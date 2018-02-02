use parenchyma::SharedTensor;
use parenchyma::error::Result;

/// An trait for dealing with transformers so that any transformable data type can be 
/// transformed into a `SharedTensor`.
pub trait Transformer {
    /// Returns the non-numeric data as a vector.
    fn as_vector(&self) -> Vec<f32>;
    /// Transforms (possibly non-numeric) data into a numeric `SharedTensor` with the provided
    /// `shape`.
    ///
    /// # Returns
    ///
    /// An `Error` is returned if the expected capacity (defined by the `shape`) differs from the
    /// observed one.
    fn transform(&self, shape: &[usize]) -> Result<SharedTensor<f32>> {
        SharedTensor::new(shape, self.as_vector())
    }
}
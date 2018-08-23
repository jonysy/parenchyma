use super::super::error::{Error, ErrorKind, Result};

/// Describes the shape of a tensor.
///
/// **note**: `From` conversion implementations are provided for low-rank shapes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorShape {
    /// The number of components the associated tensor can store.
    ///
    /// # Example
    ///
    /// ```{.text}
    /// // The following tensor has 9 components
    ///
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    pub capacity: usize,
    /// A list of numbers with each representing the dimension at each index.
    ///
    /// # Example
    ///
    /// The following tensor has a shape of `[2, 1]`:
    ///
    /// ```{.text}
    /// [[a], [b]]
    /// ```
    pub dimsizes: Vec<usize>,
    // /// The stride tells the tensor how to interpret its flattened representation.
    // stride: Vec<usize>,
}

impl TensorShape {
    /// Checks that the shape of the provided `data` is compatible.
    pub fn check<T>(&self, data: &[T]) -> Result {
        if self.capacity != data.len() {
            let message = format!(
                "TODO: incompatible shape. Capacity = {}, Length = {}", 
                self.capacity, 
                data.len());
            let kind = ErrorKind::IncompatibleShape;
            let e = Error::new(kind, message);

            return Err(e);
        }

        Ok(())
    }

    /// Returns the `dimensions`.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimsizes
    }

    /// Returns the number of elements the tensor can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the total number of indices required to identify each component uniquely (i.e, the
    /// tensor's rank, degree, or order).
    ///
    /// # Example
    ///
    /// The following tensor has a rank of 2:
    ///
    /// ```{.text}
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    pub fn rank(&self) -> usize {
        self.dimsizes.len()
    }
}

impl From<Vec<usize>> for TensorShape {

    fn from(vector: Vec<usize>) -> TensorShape {

        TensorShape {
            capacity: vector.iter().fold(1, |acc, &dims| acc * dims),
            dimsizes: vector,
        }
    }
}

impl<'slice> From<&'slice [usize]> for TensorShape {

    fn from(slice: &[usize]) -> TensorShape {
        TensorShape {
            capacity: slice.iter().fold(1, |acc, &dims| acc * dims),
            dimsizes: slice.to_owned(),
        }
    }
}

impl From<usize> for TensorShape {

    fn from(dimensions: usize) -> TensorShape {
        TensorShape {
            capacity: dimensions,
            dimsizes: vec![dimensions],
        }
    }
}

macro_rules! shape {
    ($($length:expr),*) => ($(impl From<[usize; $length]> for TensorShape {
        fn from(array: [usize; $length]) -> TensorShape {

            TensorShape {
                capacity: array.iter().fold(1, |acc, &dims| acc * dims),
                dimsizes: array.to_vec(),
            }
        }
    })*)
}

shape!(0, 1, 2, 3, 4, 5, 6);
pub struct Tensor {
	/// The total number of contravariant and covariant indices.
	///
	/// # Example
	///
	/// The following tensor has a rank of 2:
	///
	/// ```ignore
	/// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	/// ```
	pub(super) rank: usize,
	/// The number of components:
	///
	/// ```ignore
	/// dimension^(r + s) = dimension^rank
	///
	/// where r is the number of contravariant indices and s is the number of covariant indices
	/// ```
	///
	/// # Example
	///
	/// The following tensor has 9 components (dimension^rank = 3^2)
	/// ```ignore
	/// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	/// ```
	pub(super) ncomponents: usize,

	pub(super) shape: Vec<usize>,
}

impl Tensor {

	pub fn ncomponents(&self) -> usize {
		self.ncomponents
	}

	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub fn default_stride(&self) -> Vec<usize> {
		let mut strides: Vec<usize> = Vec::with_capacity(self.rank);
		let length = self.shape.len();
		match length {
			0 => strides,

			1 => {
				strides.push(1);
				strides
			},

			_ => {
				let imp = &self.shape[1..length];

				for (i, _) in imp.iter().enumerate() {
					strides.push(imp[i..imp.len()].iter().fold(1, |prod, &x| prod * x));
				}

				strides.push(1);
				strides
			}
		}
	}
}

impl<I> From<I> for Tensor where I: Into<Vec<usize>> {

	/// # Example
	///
	/// A 0-D tensor (a scalar).
	///
	/// ```rust
	/// use parenchyma::Tensor;
	///
	/// let _ = Tensor::from(vec![]);
	/// ```
	///
	/// A 1-D tensor with shape `[5]`
	///
	/// ```rust
	/// use parenchyma::Tensor;
	///
	/// let _ = Tensor::from(vec![5]);
	/// ```
	///
	/// An n-D tensor with shape `[D_1, D_2, ..., D_n]`
	///
	/// ```rust,ignore
	/// let tensor = Tensor::from(vec![D_1, D_2, ..., D_n]);
	/// ```
	fn from(shape: I) -> Tensor {

		let shape = shape.into();

		let rank = shape.len();
		let ncomponents = if rank == 0 { 1 } else { shape.iter().fold(1, |s, &a| s * a) };

		Tensor {
			rank: rank,
			ncomponents: ncomponents,
			shape: shape,
		}
	}
}
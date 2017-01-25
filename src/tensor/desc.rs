use std::ops::Deref;

pub struct TensorDesc(Vec<usize>);

impl TensorDesc {
	fn rank(&self) -> usize {
		self.len()
	}

	fn size(&self) -> usize {
		match self.rank() {
			0 => 1,
			_ => self.iter().fold(1, |s, &a| s * a)
		}
	}

	fn dims(&self) -> &[usize] {
		self
	}

	fn dims_i32(&self) -> Vec<i32> {
		self.iter().map(|&e| e as i32).collect()
	}

	fn default_stride(&self) -> Vec<usize> {
        let mut strides: Vec<usize> = Vec::with_capacity(self.rank());
        let dim_length = self.dims().len();
        match dim_length {
            0 => strides,
            1 => {
                strides.push(1);
                strides
            },
            _ => {
                let imp_dims = &self.dims()[1..dim_length];
                for (i, _) in imp_dims.iter().enumerate() {
                    strides.push(imp_dims[i..imp_dims.len()].iter().fold(1, |prod, &x| prod * x))
                }
                strides.push(1);
                strides
            }
        }
    }

	fn default_stride_i32(&self) -> Vec<i32> {
        self.default_stride().iter().map(|&e| e as i32).collect()
	}
}

impl Deref for TensorDesc {

	type Target = [usize];

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<'a> From<&'a ()> for TensorDesc {

	fn from(_: &()) -> Self {

		TensorDesc(Vec::with_capacity(1))
	}
}

impl<'a> From<&'a usize> for TensorDesc {

	fn from(n: &usize) -> Self {

		TensorDesc(vec![*n])
	}
}

impl<'a> From<&'a u32> for TensorDesc {

	fn from(n: &u32) -> Self {

		TensorDesc(vec![*n as usize])
	}
}

impl<'a> From<&'a isize> for TensorDesc {

	fn from(n: &isize) -> Self {

		TensorDesc(vec![*n as usize])
	}
}

impl<'a> From<&'a i32> for TensorDesc {

	fn from(n: &i32) -> Self {

		TensorDesc(vec![*n as usize])
	}
}

impl<'a> From<&'a Vec<usize>> for TensorDesc {

	fn from(vec: &Vec<usize>) -> Self {

		TensorDesc(vec.clone())
	}
}

impl<'a> From<&'a [usize]> for TensorDesc {

	fn from(s: &[usize]) -> Self {

		TensorDesc(s.to_owned())
	}
}

macro_rules! impl_tensor_desc_from_tuple {
	($($N:expr),*) => {
		impl<'a> From<&'a ($(usize,)*)> for TensorDesc {

			fn from(&($($N,)*): ($(usize,)*)) -> Self {
				TensorDesc(vec![$($N,)*])
			}
		}
	}
}

impl<'a> From<&'a (usize, usize)> for TensorDesc {

	fn from(&(a, b): &(usize, usize)) ->Self {

		TensorDesc(vec![a, b])
	}
}

impl<'a> From<&'a (usize, usize, usize)> for TensorDesc {

	fn from(&(a, b, c): &(usize, usize, usize)) ->Self {

		TensorDesc(vec![a, b, c])
	}
}

impl<'a> From<&'a (usize, usize, usize, usize)> for TensorDesc {

	fn from(&(a, b, c, d): &(usize, usize, usize, usize)) ->Self {

		TensorDesc(vec![a, b, c, d])
	}
}

impl<'a> From<&'a (usize, usize, usize, usize, usize)> for TensorDesc {

	fn from(&(a, b, c, d, e): &(usize, usize, usize, usize, usize)) ->Self {

		TensorDesc(vec![a, b, c, d, e])
	}
}

macro_rules! impl_tensor_desc_from_array {
	($($N: expr)+) => {

		$(

			impl<'a> From<&'a [usize; $N]> for TensorDesc {

				fn from(array: &[usize; $N]) -> Self {

					TensorDesc(From::from(&array[..]))
				}
			}
		)+
	}
}

impl_tensor_desc_from_array!(1 2 3 4 5 6);
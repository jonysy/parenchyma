use std::ops::Deref;

#[derive(Debug)]
pub struct VecExtn(Vec<usize>);

impl VecExtn {
	pub fn rank(&self) -> usize {

		self.len()
	}

	pub fn size(&self) -> usize {

		match self.rank() {
			0 => 1,
			_ => self.iter().fold(1, |s, &a| s * a)
		}
	}
}

impl Deref for VecExtn {

	type Target = Vec<usize>;

	fn deref(&self) -> &Vec<usize> {

		&self.0
	}
}

impl From<Vec<usize>> for VecExtn {

	fn from(vec: Vec<usize>) -> Self {

		VecExtn(vec)
	}
}

impl From<()> for VecExtn {

	fn from(_: ()) -> Self {

		Vec::with_capacity(1).into()
	}
}

impl From<usize> for VecExtn {

	fn from(n: usize) -> Self {

		VecExtn(vec![n])
	}
}

impl From<u32> for VecExtn {

	fn from(n: u32) -> Self {

		VecExtn(vec![n as usize])
	}
}

impl From<isize> for VecExtn {

	fn from(n: isize) -> Self {

		VecExtn(vec![n as usize])
	}
}

impl From<i32> for VecExtn {

	fn from(n: i32) -> Self {

		VecExtn(vec![n as usize])
	}
}
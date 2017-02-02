use std::{mem, slice};

/// A `Box` without any knowledge of its underlying type.
pub struct NativeMemory {
	/// The wrapped raw pointer
	raw: *mut [u8],
	len: usize,
}

impl NativeMemory {
	/// Access memory as a slice.
	pub fn as_slice<T>(&self) -> &[T] {

		unsafe {

			slice::from_raw_parts_mut(self.raw as *mut T, self.len / mem::size_of::<T>())
		}
	}

	/// Access memory as a mutable slice.
	pub fn as_mut_slice<T>(&self) -> &mut [T] {

		unsafe {

			slice::from_raw_parts_mut(self.raw as *mut T, self.len / mem::size_of::<T>())
		}
	}
}

impl From<Box<[u8]>> for NativeMemory {

	fn from(b: Box<[u8]>) -> NativeMemory {

		NativeMemory { len: b.len(), raw: Box::into_raw(b) }
	}
}

impl Drop for NativeMemory {
	fn drop(&mut self) {

		unsafe {

			let _ = Box::from_raw(self.raw);
		}
	}
}
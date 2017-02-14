use alloc::raw_vec::RawVec;

/// An alternative way to allocate memory, but requires [`RawVec`][RawVec] which is currently
/// unstable (*[#27783]).
///
/// [#27783]: https://github.com/rust-lang/rust/issues/27783
/// [RawVec]: https://doc.rust-lang.org/alloc/raw_vec/struct.RawVec.html
pub fn allocate_boxed_slice(capacity: usize) -> Box<[u8]> {
	let raw_vec = RawVec::with_capacity(capacity);

	unsafe {
		raw_vec.into_box()
	}
}
/// Traditional allocation through the creation of a filled `Vec<u8>` of length `capacity`.
pub fn allocate_boxed_slice(capacity: usize) -> Box<[u8]> {
	let vec: Vec<u8> = vec![0; capacity];
	vec.into_boxed_slice()
}
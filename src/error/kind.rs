use super::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
	/// A framework-specific error.
	Framework { name: &'static str },
	/// `BitMap` (currently a type alias for `u64`) capacity has been reached.
	BitMapCapacityExceeded,
	/// Invalid reshaped tensor size.
	InvalidReshapedTensorSize,
	/// An error returned when attempting to access uninitialized memory.
	UninitializedMemory,
	/// Memory allocation was not found for a provided `Context`.
	AllocatedMemoryNotFoundForContext,
	/// An error occurred while attempting to synchronize memory.
	MemorySynchronizationFailed,
	/// A memory synchronization route was requested, but no available synchronization route was found.
	NoAvailableSynchronizationRouteFound,
	/// An error occurred while attempting to allocate memory.
	MemoryAllocationFailed,
	/// Any error not part of this list.
	Other,
    /// A marker variant that tells the compiler that users of this enum cannot match 
    /// it exhaustively ([related RFC](https://github.com/rust-lang/rust/issues/32770)).
    #[doc(hidden)]
    _NonExhaustive,
}

impl ErrorKind {

	pub(super) fn as_str(&self) -> &'static str {

		use self::ErrorKind::*;

		match *self {
			Framework { name } => name,
			BitMapCapacityExceeded => "the maximum number of backing memories has been reached",
			InvalidReshapedTensorSize => "size of the provided shape is not equal to the size of the current shape",
			UninitializedMemory => "uninitialized memory",
			AllocatedMemoryNotFoundForContext => "memory allocation was not found for the provided context",
			MemorySynchronizationFailed => "memory synchronization failed",
			NoAvailableSynchronizationRouteFound => "no available memory synchronization route",
			MemoryAllocationFailed => "memory allocation failed",
			Other => "other error",
			_ => unreachable!(),
		}
	}
}

impl From<ErrorKind> for Error {

	/// Creates a new error from a known kind of error
	fn from(kind: ErrorKind) -> Error {

		Error::_new(kind, None)
	}
}
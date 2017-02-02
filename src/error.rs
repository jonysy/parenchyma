use std::error;

#[derive(Debug)]
pub enum ErrorKind {
	Memory,
	MissingDestination,
	InvalidMemoryAllocation,

	/// A marker variant that tells the compiler that users of this enum cannot match it exhaustively.
	#[doc(hidden)]
	__Nonexhaustive,
}

#[derive(Debug)]
pub struct Error {
	kind: ErrorKind,
	error: Box<error::Error>,
}

macro_rules! error_fn {
	($fun: ident, $kind: expr) => {

		pub fn $fun<E>(error: E) -> Error where E: Into<Box<error::Error>> {

			Error::new($kind, error)
		}
	}
}

impl Error {

	pub fn new<E: Into<Box<error::Error>>>(kind: ErrorKind, e: E) -> Error {

		Error {
			kind: kind,
			error: e.into()
		}
	}

	error_fn!(memory, ErrorKind::Memory);
	error_fn!(missing_destination, ErrorKind::MissingDestination);
	error_fn!(invalid_memory_allocation, ErrorKind::InvalidMemoryAllocation);
}
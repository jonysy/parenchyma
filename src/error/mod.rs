pub use self::kind::ErrorKind;
pub use self::result::Result;

mod kind;
mod result;

// ==============

use std::{error, fmt};
use std::ops::Deref;

#[derive(Debug)]
pub struct Error {
	kind: ErrorKind,
	payload: Option<Box<error::Error + Send + Sync>>,
}

impl Error {

	/// Creates a new error from a known kind of error as well as an arbitrary error payload.
	pub fn new<K, E>(kind: K, payload: E) -> Error 
		where K: Into<ErrorKind>, 
			  E: Into<Box<error::Error + Send + Sync>>
	{

		Self::_new(kind.into(), Some(payload.into()))
	}

	// "De-generization" technique..
	fn _new(kind: ErrorKind, payload: Option<Box<error::Error + Send + Sync>>) -> Error {

		Error {
			kind: kind,
			payload: payload
		}
	}

	pub fn get_ref(&self) -> Option<&(error::Error + Send + Sync + 'static)> {

		match self.payload {
			Some(ref payload) => Some(payload.deref()),
			_ => None
		}
	}

	/// Returns the corresponding `ErrorKind` for this error.
	pub fn kind(&self) -> ErrorKind {
		self.kind
	}
}

impl fmt::Display for Error {

	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {

		write!(fmt, "{}", self.kind.as_str())
	}
}

impl error::Error for Error {

	fn description(&self) -> &str {

		if let Some(ref payload) = self.payload {
			payload.description()
		} else {
			self.kind.as_str()
		}
	}

	fn cause(&self) -> Option<&error::Error> {

		match self.payload {
			Some(ref payload) => {
				payload.cause()
			},
			_ => {
				None
			}
		}
	}
}

#[cfg(test)]
mod test {
	use super::{Error, ErrorKind};
	use std::{error, fmt};

	#[test]
	fn test_downcasting() {
		#[derive(Debug)]
		struct TestError;
		impl fmt::Display for TestError { fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
			Ok(())
		}}
		impl error::Error for TestError { fn description(&self) -> &str { "abc" } }
		let err = Error::new(ErrorKind::Other, TestError);
		assert!(err.get_ref().unwrap().is::<TestError>());
		assert_eq!("abc", err.get_ref().unwrap().description());
	}
}
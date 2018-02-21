//! Types for working with errors.

use std::{error, fmt};
use std::ops::Deref;

/// A specialized `Result` typedef.
pub type Result<T = (), E = Error> = ::std::result::Result<T, E>;

/// The error structure used by the Parenchyma crate.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    /// A boxed sendable, syncable `Error`.
    inner: Option<Box<::std::error::Error + Send + Sync>>,
}

/// A set of general categories.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// A framework-specific error.
    ///
    /// Consider creating an framework-specific error by calling the `Error::from_framework` 
    /// function, rather than constructing an `Error` using this variant.
    Framework(&'static str),
    /// An error returned when attempting to access uninitialized memory.
    UninitializedMemory,
    /// Unable to drop the provided device because a memory allocation was not found for it.
    AllocatedMemoryNotFoundForDevice,
    /// An error occurred while attempting to synchronize memory.
    MemorySynchronizationFailed,
    /// A memory synchronization route was requested, but no available synchronization route was found.
    NoAvailableSynchronizationRouteFound,
    /// An error occurred while attempting to allocate memory.
    MemoryAllocationFailed,
    /// An error occurred while downcasting
    MemoryDowncasting,

    // MARK: - A set of tensor error categories

    /// Maximum number of backing memories has been reached (`BitMap` - type alias for `u64`).
    CapacityExceeded,
    /// The tensor shape is incompatible with the shape of some data.
    IncompatibleShape,
    /// Invalid reshaped tensor size.
    InvalidReshapedTensorSize,

    /// Any error not part of this list.
    Other,
    /// A marker variant that tells the compiler that users of this enum cannot match 
    /// it exhaustively ([related RFC](https://github.com/rust-lang/rust/issues/32770)).
    #[doc(hidden)]
    _NonExhaustive,
}

impl ErrorKind {
    fn as_str(&self) -> &'static str {
        use self::ErrorKind::*;

        match *self {
            Framework(name) => name,
            CapacityExceeded => "the maximum number of backing memories has been reached",
            IncompatibleShape => "the tensor shape is incompatible with the shape of the data",
            InvalidReshapedTensorSize => "size of the provided shape is not equal to the size of the current shape",
            UninitializedMemory => "uninitialized memory",
            AllocatedMemoryNotFoundForDevice => "memory allocation was not found for the provided device",
            MemorySynchronizationFailed => "memory synchronization failed",
            NoAvailableSynchronizationRouteFound => "no available memory synchronization route",
            MemoryAllocationFailed => "memory allocation failed",
            MemoryDowncasting => "something went wrong while downcasting",
            Other => "other error",
            _ => unreachable!(),
        }
    }
}

impl Error {
    /// Creates a new error from a known kind of error as well as an arbitrary error error.
    pub fn new<K, E>(kind: K, error: E) -> Error 
        where K: Into<ErrorKind>, 
              E: Into<Box<error::Error + Send + Sync>> {
        Self::_new(kind.into(), Some(error.into()))
    }

    /// Returns a reference to the inner error wrapped by this error (if any).
    pub fn get_ref(&self) -> Option<&(error::Error + Send + Sync + 'static)> {
        match self.inner {
            Some(ref error) => Some(error.deref()),
            _ => None
        }
    }
    /// Returns the corresponding `ErrorKind` for this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl Error {
    // "De-generization" technique..
    fn _new(kind: ErrorKind, error: Option<Box<error::Error + Send + Sync>>) -> Error {
        Error { kind, inner: error }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.kind.as_str())
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        if let Some(ref error) = self.inner {
            error.description()
        } else {
            self.kind.as_str()
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match self.inner {
            Some(ref error) => error.cause(),
            _ => None,
        }
    }
}

impl From<ErrorKind> for Error {
    /// Creates a new error from a known kind of error
    fn from(kind: ErrorKind) -> Error {
        Error::_new(kind, None)
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
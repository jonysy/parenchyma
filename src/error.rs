use std::{error, fmt, result};

/// A specialized `Result` type.
pub type Result<T = ()> = result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    payload: Option<Box<error::Error + Send + Sync>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// A framework-specific error.
    Framework { name: &'static str },
    /// Maximum number of backing memories has been reached (`BitMap` - type alias for `u64`).
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
        use std::ops::Deref;

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
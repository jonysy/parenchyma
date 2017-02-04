use std::result;
use super::Error;

/// A specialized `Result` type.
pub type Result<T = ()> = result::Result<T, Error>;
//! Useful data type definitions

use error::Error;
use futures::Future;
use memory::Chunk;

/// The success type returned by allocator methods.
pub type BoxChunk = Box<Chunk>;

/// Unlike the type alias provided by `futures` crate, this typedef is specialized for use within
/// Parenchyma.
///
/// **note**:
///
/// The `Future` must outlive the provided lifetime.
pub type BoxFuture<'a, T: 'a = ()> = Box<Future<Item=T, Error=Error> + 'a>;

/// A specialized `Result` type.
pub type Result<T = (), E = Error> = ::std::result::Result<T, E>;
//! Helper functions and traits

/// Marker trait for uninitialized objects.
pub type Uninitialized = ();

/// A trait for simple _has_ relationships.
pub trait Has<T: ?Sized> {

    /// Get a reference to `T`.
    fn get_ref(&self) -> &T;
}

/// Attempt to construct a default value of a type.
// TODO move to a crate?
pub trait TryDefault: Sized {
    /// The type returned in the event of an error.
    type Err;

    /// Returns the default value for a type.
    fn try_default() -> Result<Self, Self::Err>;
}

/// Returns the size of the allocated memory in bytes.
pub fn allocated<T>(length: usize) -> usize {
    use std::mem;

    length * mem::size_of::<T>()
}
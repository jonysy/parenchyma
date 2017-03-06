use std::mem;

/// Returns the size of the allocated memory in bytes.
pub fn allocated<T>(length: usize) -> usize {
    length * mem::size_of::<T>()
}

pub unsafe fn extend_lifetime<'a, 'b, T>(t: &'a T) -> &'b T {
    mem::transmute::<&'a T, &'b T>(t)
}

unsafe fn extend_lifetime_mut<'a, 'b, T>(t: &'a mut T) -> &'b mut T {
    mem::transmute::<&'a mut T, &'b mut T>(t)
}
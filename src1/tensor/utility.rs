use std::mem;

pub unsafe fn extend_lifetime<'a, 'b, T>(t: &'a T) -> &'b T {
    mem::transmute::<&'a T, &'b T>(t)
}

pub unsafe fn extend_lifetime_mut<'a, 'b, T>(t: &'a mut T) -> &'b mut T {
    mem::transmute::<&'a mut T, &'b mut T>(t)
}
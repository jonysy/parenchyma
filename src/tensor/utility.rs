use std::mem;

pub(in super) unsafe fn extend_lifetime<'a, 'b, T>(t: &'a T) -> &'b T 
    where T: ?Sized {

    mem::transmute::<&'a T, &'b T>(t)
}

pub(in super) unsafe fn extend_lifetime_mut<'a, 'b, T>(t: &'a mut T) -> &'b mut T 
    where T: ?Sized {

    mem::transmute::<&'a mut T, &'b mut T>(t)
}
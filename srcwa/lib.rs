#![feature(generic_associated_types)]

pub trait FrameworkCtr {
    type Context: ContextCtor;
}

pub trait ContextCtor {
    type Package;
}d
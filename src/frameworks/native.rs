use parenchyma::frameworks::NativeContext as Context;
use super::super::{Dependencies, Extension};

impl<P> Extension for Context<P> where P: Dependencies { }
use std::any::Any;
use super::super::ObjectSafeContext;

pub struct Location { pub(super) context: Box<ObjectSafeContext>, pub(super) memory: Box<Any> }
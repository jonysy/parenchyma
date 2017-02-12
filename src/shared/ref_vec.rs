use std::any::Any;
use std::cell::RefCell;
use super::super::context::ContextView;

pub struct RVec { pub re: RefCell<Vec<Location>> }

pub struct Location { pub context: Box<ContextView>, pub memory: Box<Any> }

impl RVec {

    pub fn new() -> RVec {
        RVec {
            re: RefCell::new(vec![]),
        }
    }

    pub fn position<P>(&self, predicate: P) -> Option<usize> where P: FnMut(&Location) -> bool {
        self.re.borrow().iter().position(predicate)
    }

    pub fn push(&self, context: Box<ContextView>, memory: Box<Any>) {
        self.re.borrow_mut().push(Location { context: context, memory: memory })
    }

    pub fn len(&self) -> usize {
        self.re.borrow().len()
    }

    pub fn remove(&self, index: usize) -> Location {
        self.re.borrow_mut().remove(index)
    }
}
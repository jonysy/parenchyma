use std::any::Any;
use std::cell::RefCell;
use super::super::device::ObjSafeDev;

#[derive(Debug)]
pub struct RVec { pub re: RefCell<Vec<Location>> }

#[derive(Debug)]
pub struct Location { pub device: Box<ObjSafeDev>, pub memory: Box<Any> }

impl RVec {

    pub fn new() -> RVec {
        RVec {
            re: RefCell::new(vec![]),
        }
    }

    pub fn position<P>(&self, predicate: P) -> Option<usize> where P: FnMut(&Location) -> bool {
        self.re.borrow().iter().position(predicate)
    }

    pub fn push(&self, device: Box<ObjSafeDev>, memory: Box<Any>) {
        self.re.borrow_mut().push(Location { device: device, memory: memory })
    }

    pub fn len(&self) -> usize {
        self.re.borrow().len()
    }

    pub fn remove(&self, index: usize) -> Location {
        self.re.borrow_mut().remove(index)
    }
}
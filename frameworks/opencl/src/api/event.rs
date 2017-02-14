use super::sys;

pub struct Event(sys::cl_event);

impl From<sys::cl_event> for Event {
    
    fn from(cl_event: sys::cl_event) -> Self {
        
        Event(cl_event)
    }
}
use hardware::Hardware;
use utility::typedef::Result;

pub trait Context {

    fn from(hardware: Vec<Hardware>) -> Result<Self> 
        where Self: Sized {

        unimplemented!()
    }
}
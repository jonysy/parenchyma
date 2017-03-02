// pub trait Framework {

//     fn devices(&self) -> &[Box<Device>];
// }

// pub trait Device: Eq + PartialEq { 

//     fn allocate_memory(&mut self, size: usize) -> Result;

//     fn synch_in(&mut self, buffer: &Buffer) -> Result;

//     fn synch_out(&self, buffer: &mut Buffer) -> Result;
// }
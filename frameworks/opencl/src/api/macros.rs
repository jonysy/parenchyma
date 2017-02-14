macro_rules! result {
    
    ($exp:expr => $ok:expr) => {
        match $exp {
            ::api::sys::CLStatus::CL_SUCCESS => $ok,
            
            error @ _ => {
                let error_kind: ::api::error::ErrorKind = error.into();
                Err(error_kind.into())
            }
        }
    };
    
    ($exp:expr) => {
        result!($exp => Ok(()))
    };
}
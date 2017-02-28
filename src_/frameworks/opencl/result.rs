use std::result;
use super::api::error::Error;

pub type Result<T = ()> = result::Result<T, Error>;
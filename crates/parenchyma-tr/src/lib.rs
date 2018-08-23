//! A framework for pre and post processing machine intelligence based data

extern crate image;
extern crate parenchyma;

pub use self::transformer::Transformer;

mod transformer;
mod transformers;
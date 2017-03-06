//! Native backend support.

use std::{cmp, mem, slice};

/// Provides the native framework.
#[derive(Debug)]
pub struct Native;

/// The native context.
#[derive(Clone, Debug)]
pub struct NativeContext;

/// The native device.
#[derive(Clone, Debug)]
pub struct NativeDevice;

impl cmp::PartialEq for NativeDevice {

    fn eq(&self, _: &Self) -> bool { true }
}

impl cmp::Eq for NativeDevice { }
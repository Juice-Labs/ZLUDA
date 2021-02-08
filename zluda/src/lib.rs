#![allow(dead_code)]

#[macro_use]
extern crate lazy_static;
#[cfg(test)]
extern crate cuda_driver_sys;
#[cfg(test)]
#[macro_use]
extern crate paste;
extern crate ptx;

#[allow(warnings)]
pub mod cuda;
mod cuda_impl;
pub(crate) mod r#impl;

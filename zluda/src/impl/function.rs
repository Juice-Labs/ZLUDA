use super::{stream::Stream, CUresult, GlobalState, HasLivenessCookie, LiveCheck};
use crate::cuda::CUfunction_attribute;
use ::std::os::raw::{c_uint, c_void};
use std::{hint, ptr};

const CU_LAUNCH_PARAM_END: *mut c_void = 0 as *mut _;
const CU_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 1 as *mut _;
const CU_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 2 as *mut _;

pub type Function = LiveCheck<FunctionData>;

impl HasLivenessCookie for FunctionData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0x5e2ab14d5840678e;

    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0x33e6a1e6;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        Ok(())
    }
}

pub struct FunctionData {
    pub base: c_uint,
    pub arg_size: Vec<usize>,
    pub use_shared_mem: bool,
    pub legacy_args: LegacyArguments,
}

pub struct LegacyArguments {
    block_shape: Option<(i32, i32, i32)>,
}

impl LegacyArguments {
    pub fn new() -> Self {
        LegacyArguments { block_shape: None }
    }

    #[allow(dead_code)]
    pub fn is_initialized(&self) -> bool {
        self.block_shape.is_some()
    }

    pub fn reset(&mut self) {
        self.block_shape = None;
    }
}

impl FunctionData {
    fn get_properties(&mut self) -> Result<c_uint, c_uint> {
        Err(0)
    }
}

pub fn launch_kernel(
    f: *mut Function,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    hstream: *mut Stream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<(), CUresult> {
    if f == ptr::null_mut()
        || (kernel_params == ptr::null_mut() && extra == ptr::null_mut())
        || (kernel_params != ptr::null_mut() && extra != ptr::null_mut())
    {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock_stream(hstream, |stream| {
        let func: &mut FunctionData = unsafe { &mut *f }.as_result_mut()?;
        if kernel_params != ptr::null_mut() {
        } else {
            let mut offset = 0;
            let mut buffer_ptr = None;
            let mut buffer_size = None;
            loop {
                match unsafe { *extra.add(offset) } {
                    CU_LAUNCH_PARAM_END => break,
                    CU_LAUNCH_PARAM_BUFFER_POINTER => {
                        buffer_ptr = Some(unsafe { *extra.add(offset + 1) as *mut u8 });
                    }
                    CU_LAUNCH_PARAM_BUFFER_SIZE => {
                        buffer_size = Some(unsafe { *(*extra.add(offset + 1) as *mut usize) });
                    }
                    _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
                }
                offset += 2;
            }
            match (buffer_size, buffer_ptr) {
                (Some(buffer_size), Some(buffer_ptr)) => {
                    let sum_of_kernel_argument_sizes =
                        func.arg_size.iter().fold(0, |offset, size_of_arg| {
                            size_of_arg + round_up_to_multiple(offset, *size_of_arg)
                        });
                    if buffer_size != sum_of_kernel_argument_sizes {
                        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
                    }
                    let mut offset = 0;
                    for (i, arg_size) in func.arg_size.iter().enumerate() {
                        let buffer_offset = round_up_to_multiple(offset, *arg_size);
                        offset = buffer_offset + *arg_size;
                    }
                }
                _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
            }
        }
        Ok(())
    })?
}

fn round_up_to_multiple(x: usize, multiple: usize) -> usize {
    ((x + multiple - 1) / multiple) * multiple
}

pub(crate) fn get_attribute(
    pi: *mut i32,
    attrib: CUfunction_attribute,
    func: *mut Function,
) -> Result<(), CUresult> {
    if pi == ptr::null_mut() || func == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    Err(CUresult::CUDA_ERROR_NOT_SUPPORTED)
}

pub(crate) fn set_block_shape(func: *mut Function, x: i32, y: i32, z: i32) -> Result<(), CUresult> {
    if func == ptr::null_mut() || x < 0 || y < 0 || z < 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock_function(func, |func| {
        func.legacy_args.block_shape = Some((x, y, z));
    })
}

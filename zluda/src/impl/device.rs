use super::{context, CUresult, GlobalState};
use crate::cuda;
use cuda::{CUdevice_attribute, CUuuid_st};
use std::{
    cmp, mem,
    os::raw::{c_char, c_int, c_uint},
    ptr,
    sync::atomic::{AtomicU32, Ordering},
};

const PROJECT_URL_SUFFIX_SHORT: &'static str = " [ZLUDA]";
const PROJECT_URL_SUFFIX_LONG: &'static str = " [github.com/vosen/ZLUDA]";

#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Index(pub c_int);

pub struct Device {
    pub index: Index,
    pub base: c_uint,
    pub default_queue: c_uint,
    pub l0_context: c_uint,
    pub primary_context: context::Context,
    properties: Option<Box<l0::sys::ze_device_properties_t>>,
    image_properties: Option<Box<l0::sys::ze_device_image_properties_t>>,
    memory_properties: Option<Vec<l0::sys::ze_device_memory_properties_t>>,
    compute_properties: Option<Box<l0::sys::ze_device_compute_properties_t>>,
}

unsafe impl Send for Device {}

impl Device {
    // Unsafe because it does not fully initalize primary_context
    unsafe fn new(drv: c_uint, l0_dev: c_uint, idx: usize) -> Result<Self, CUresult> {
        let mut ctx = 0;
        let queue = 0;
        let primary_context = context::Context::new(context::ContextData::new(
            drv,
            l0_dev,
            0,
            true,
            ptr::null_mut(),
        )?);
        Ok(Self {
            index: Index(idx as c_int),
            base: l0_dev,
            default_queue: queue,
            l0_context: ctx,
            primary_context: primary_context,
            properties: None,
            image_properties: None,
            memory_properties: None,
            compute_properties: None,
        })
    }

    fn get_properties<'a>(&'a mut self) -> l0::Result<&'a l0::sys::ze_device_properties_t> {
        if let Some(ref prop) = self.properties {
            return Ok(prop);
        }
        match self.base.get_properties() {
            Ok(prop) => Ok(self.properties.get_or_insert(prop)),
            Err(e) => Err(e),
        }
    }

    fn get_image_properties(&mut self) -> l0::Result<&l0::sys::ze_device_image_properties_t> {
        if let Some(ref prop) = self.image_properties {
            return Ok(prop);
        }
        match self.base.get_image_properties() {
            Ok(prop) => Ok(self.image_properties.get_or_insert(prop)),
            Err(e) => Err(e),
        }
    }

    fn get_memory_properties(&mut self) -> l0::Result<&[l0::sys::ze_device_memory_properties_t]> {
        if let Some(ref prop) = self.memory_properties {
            return Ok(prop);
        }
        match self.base.get_memory_properties() {
            Ok(prop) => Ok(self.memory_properties.get_or_insert(prop)),
            Err(e) => Err(e),
        }
    }

    fn get_compute_properties(&mut self) -> l0::Result<&l0::sys::ze_device_compute_properties_t> {
        if let Some(ref prop) = self.compute_properties {
            return Ok(prop);
        }
        match self.base.get_compute_properties() {
            Ok(prop) => Ok(self.compute_properties.get_or_insert(prop)),
            Err(e) => Err(e),
        }
    }

    pub fn late_init(&mut self) {
        self.primary_context.as_option_mut().unwrap().device = self as *mut _;
    }

    fn get_max_simd(&mut self) -> l0::Result<u32> {
        let props = self.get_compute_properties()?;
        Ok(*props.subGroupSizes[0..props.numSubGroupSizes as usize]
            .iter()
            .max()
            .unwrap())
    }
}

pub fn init(driver: c_uint) -> Result<Vec<Device>, CUresult> {
    let ze_devices : Vec<Device>;
    Ok(ze_devices)
}

pub fn get_count(count: *mut c_int) -> Result<(), CUresult> {
    let len = GlobalState::lock(|state| state.devices.len())?;
    unsafe { *count = len as c_int };
    Ok(())
}

pub fn get(device: *mut Index, ordinal: c_int) -> Result<(), CUresult> {
    if device == ptr::null_mut() || ordinal < 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let len = GlobalState::lock(|state| state.devices.len())?;
    if ordinal < (len as i32) {
        unsafe { *device = Index(ordinal) };
        Ok(())
    } else {
        Err(CUresult::CUDA_ERROR_INVALID_VALUE)
    }
}

pub fn get_name(name: *mut c_char, len: i32, dev_idx: Index) -> Result<(), CUresult> {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
}

pub fn total_mem_v2(bytes: *mut usize, dev_idx: Index) -> Result<(), CUresult> {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
}

impl CUdevice_attribute {
    fn get_static_value(self) -> Option<i32> {
        match self {
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP => Some(1),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => Some(1),
            // TODO: fix this for DG1
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => Some(1),
            // TODO: go back to this once we have more funcitonality implemented
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => Some(8),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => Some(0),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => Some(1),
            _ => None,
        }
    }
}

pub fn get_attribute(
    pi: *mut i32,
    attrib: CUdevice_attribute,
    dev_idx: Index,
) -> Result<(), CUresult> {
    return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
}

pub fn get_uuid(uuid: *mut CUuuid_st, dev_idx: Index) -> Result<(), CUresult> {
    Ok(())
}

// TODO: add support if Level 0 exposes it
pub fn get_luid(luid: *mut c_char, dev_node_mask: *mut c_uint, _dev_idx: Index) -> Result<(), CUresult> {
    unsafe { ptr::write_bytes(luid, 0u8, 8) };
    unsafe { *dev_node_mask = 0 };
    Ok(())
}

pub fn primary_ctx_get_state(
    dev_idx: Index,
    flags: *mut u32,
    active: *mut i32,
) -> Result<(), CUresult> {
    Ok(())
}

pub fn primary_ctx_retain(
    pctx: *mut *mut context::Context,
    dev_idx: Index,
) -> Result<(), CUresult> {
    let ctx_ptr = GlobalState::lock_device(dev_idx, |dev| &mut dev.primary_context as *mut _)?;
    unsafe { *pctx = ctx_ptr };
    Ok(())
}

// TODO: allow for retain/reset/release of primary context
pub(crate) fn primary_ctx_release_v2(_dev_idx: Index) -> CUresult {
    CUresult::CUDA_SUCCESS
}

#[cfg(test)]
mod test {
    use super::super::test::CudaDriverFns;
    use super::super::CUresult;

    cuda_driver_test!(primary_ctx_default_inactive);

    fn primary_ctx_default_inactive<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut flags = u32::max_value();
        let mut active = i32::max_value();
        assert_eq!(
            T::cuDevicePrimaryCtxGetState(0, &mut flags, &mut active),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
    }
}

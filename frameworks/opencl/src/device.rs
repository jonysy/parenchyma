use super::{cl, core};

#[derive(Clone, Debug)]
pub struct Device {
    pub(super) id: core::DeviceId,
    /// maximum compute units
    pub compute_units: u32,
    /// The name of the device
    pub name: String,
    /// The maximum work-group size.
    pub workgroup_size: usize,
}

impl<'d> From<&'d cl::Device> for Device {

    fn from(cl_device: &cl::Device) -> Self {

        let id = *cl_device.as_core();
        let compute_units = {
            match cl_device.info(cl::enums::DeviceInfo::MaxComputeUnits) {
                cl::enums::DeviceInfoResult::MaxComputeUnits(n) => n,
                _ => unreachable!()
            }
        };
        let name = cl_device.name();
        let workgroup_size = cl_device.max_wg_size().unwrap();

        Device {
            id: id,
            compute_units: compute_units,
            name: name,
            workgroup_size: workgroup_size,
        }
    }
}
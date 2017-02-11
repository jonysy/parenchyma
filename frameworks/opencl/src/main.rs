extern crate opencl;

use opencl::api;

fn main() {

    for platform in api::platform_ids().unwrap() {
        // println!("{}", platform.profile().unwrap());
        // println!("{}", platform.name().unwrap());
        // println!("{}", platform.vendor().unwrap());
        // println!("{:#?}", platform.extensions().unwrap());

        for device in platform.all_device_ids().unwrap() {
            print!("\n\n");

            println!("max_compute_units: {}", device.max_compute_units().unwrap());
            println!("name: {:#?}", device.name().unwrap());
            println!("type: {:#?}", device.type_().unwrap());
            println!("vendor: {:#?}", device.vendor().unwrap());
            println!("vendor_id: {:#?}", device.vendor_id().unwrap());
            println!("version: {:#?}", device.version().unwrap());
            println!("driver_version: {:#?}", device.driver_version().unwrap());
        }
    }
}
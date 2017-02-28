use Tensor;
use super::OpenCLContext;

pub fn sigmoid(ctx: &OpenCLContext, a: &Tensor<f32>) {

    let device = ::DeviceView::OpenCL(ctx.devices()[0].clone());

    let ref k = ctx.kernels.sigmoid;

    //{
        let a_size = a.mem_size();
        //let result_size = result.mem_size();

        let a_mem = a.read(&device).unwrap();
        let a_mem = a_mem.as_opencl().unwrap();

        // let result_mem = result.read_write(&device).unwrap();
        // let result_mem = result_mem.as_opencl().unwrap();

        k.set_arg(0, a_size, &a_mem.obj);
        //k.set_arg(1, result_size, &result_mem.obj);
    //}

    // add to device
    let event = device.as_opencl().unwrap().queue.as_ref().unwrap().ptr
        .enqueue_nd_range_kernel(k, a.shape(), &[], &[]).unwrap();

    // device.synch_out()
}
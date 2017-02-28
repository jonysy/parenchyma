// sigmoid
__kernel void array_sigmoid_f32(__global float *a) {
    uintptr_t i = get_global_id(0);
    a[i] = 1.0 / (1.0 + exp(1.0));
}
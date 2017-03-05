__kernel void array_sigmoid_f32(__global float *a, __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = 1.0 / (1.0 + exp(-a[i]));
}
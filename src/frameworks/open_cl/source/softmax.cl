
// TODO write a parallel reduction: 
//
// * find the max element in the array
// * ..

#define SOFTMAX(interfn) \
__kernel __attribute((reqd_work_group_size(1, 1, 1))) \
void interfn##_float(__global float* x, __global float* result, const uintptr_t len) { \
    float in_max = -MAXFLOAT; \
    float sum = 0.0; \
    uintptr_t i; \
    for(i = 0; i < len; i++) { \
        float current = x[i]; \
        in_max = (in_max > current) ? in_max : current; \
    } \
    for(i = 0; i < len; i++) { \
        float current = exp(x[i] - in_max); \
        sum += current; \
        result[i] = current; \
    } \
    for(i = 0; i < len; i++) { \
        result[i] = interfn(result[i] / sum); \
    } \
} \

#define softmax(x) (x)
SOFTMAX(softmax)

#define log_softmax(x) (log(x))
SOFTMAX(log_softmax)

__kernel void log_softmax_backward_float(
    __global float* x, 
    __global float* x_diff, 
    __global float* result,
    const uintptr_t len)
{
    float sum = 0.0;
    uintptr_t i;
    for(i = 0; i < len; i++) {
        sum += x_diff[i];
    }
    for(i = 0; i < len; i++) {
        result[i] = x_diff[i] - exp(x[i]) * sum;
    }
}
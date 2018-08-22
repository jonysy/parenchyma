#define ACTIVATION_TYPE(function, type) \
kernel void function##_##type(global const type* in, global type* out, const uintptr_t len) \
{ \
    const uintptr_t current = get_global_id(0); \
    if(current >= len) { \
        return void(); \
    } \
    out[current] = function(in[current]); \
} \

#define ACTIVATION(function) ACTIVATION_TYPE(function, float) ACTIVATION_TYPE(function, double) \

// =================================================================================================

ACTIVATION(tanh)

#define sigmoid(x) (1 / (1 + exp(-x)))
ACTIVATION(sigmoid)

#define relu(x) (x > 0 ? x : 0)
ACTIVATION(relu)

#define elu(x) (x > 0 ? x : exp(x) - 1)
ACTIVATION(elu)
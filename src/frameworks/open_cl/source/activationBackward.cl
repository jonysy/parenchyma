
// TODO newline required for some reason..
#define BACKWARD_WITH_TYPE(name, type, activationDeriv) \
kernel void name##_backward_##type(global const type* in, global const type* inDiff, global type* outDiff, const uintptr_t len) \
{ \
    const uintptr_t current = get_global_id(0); \
    if(current >= len) { \
        return void(); \
    } \
    outDiff[current] = activationDeriv(in[current]) * inDiff[current]; \
} \

#define BACKWARD(name, deriv) \ 
BACKWARD_WITH_TYPE(name, float, deriv) BACKWARD_WITH_TYPE(name, double, deriv) \

// =================================================================================================

#define tanhDeriv(x) (1 - x * x)
BACKWARD(tanh, tanhDeriv)

#define sigmoidDeriv(x) (x * (1 - x))
BACKWARD(sigmoid, sigmoidDeriv)

#define reluDeriv(x) (x > 0 ? 1 : 0)
BACKWARD(relu, reluDeriv)

#define eluDeriv(x) (x > 0 ? 1 : x + 1)
BACKWARD(elu, eluDeriv)
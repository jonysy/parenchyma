# parenchyma-deep

This package provides full NN support for Parenchyma, so you can use NN on servers, desktops or 
mobiles, GPUs, FPGAs or CPUS, without worrying about OpenCL or CUDA support on the machine.

## Provided Operations

This package provides the following operations to Parenchyma backends:

|                       | CUDA (cuDNN)  | OpenCL    | Native (rust) |
|---                    |---            |---        |---            |
| Sigmoid               | (collenchyma) | -         | ✓             |
| Sigmoid (pointwise)   | (collenchyma) | -         |               |
| ReLU                  | (collenchyma) | -         | ✓             |
| ReLU (pointwise)      | (collenchyma) | -         |               |
| Tanh                  | (collenchyma) | -         | ✓             |
| Tanh (pointwise)      | (collenchyma) | -         |               |
|                       |               |           |               |
| Normalization (LRN)   | (collenchyma) | -         | -             |
|                       |               |           |               |
| Convolution           | (collenchyma) | -         | -             |
|                       |               |           |               |
| Softmax               | (collenchyma) | -         | ✓             |
| Log Softmax           | (collenchyma) | -         | ✓             |
|                       |               |           |               |
| Pooling Max           | (collenchyma) | -         | -             |
| Pooling Avg           | (collenchyma) | -         | -             |
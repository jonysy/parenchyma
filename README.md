# parenchyma-blas

This package provides full BLAS (Basic Linear Algebra Subprograms) support for Parenchyma, so you 
can use BLAS on servers, desktops or mobiles, GPUs, FPGAs or CPUS, without worrying about OpenCL or 
CUDA support on the machine.

## Provided Operations

This package provides the following operations to Parenchyma backends:

|           | CUDA (cuBLAS) | OpenCL    | Native (rblas)    |
|---        |---            |---        |---                |
| Level 1   | (collenchyma) | ✓         | ✓                 |
| Level 2   | -             | -         | -                 |
| Level 3   | (collenchyma) | (some)    | (some)            | 
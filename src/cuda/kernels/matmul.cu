extern "C" {

#include <stdio.h>

// Stub implementation that will be replaced with actual CUDA kernel
__attribute__((visibility("default")))
int matmul_kernel(
    const void* A,
    const void* B,
    void* C,
    unsigned int rowsA,
    unsigned int colsA,
    unsigned int colsB)
{
    return 0; // success
}

} // extern "C"
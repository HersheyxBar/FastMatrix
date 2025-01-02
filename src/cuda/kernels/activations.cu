extern "C" {
#include <stdio.h>

// Stub implementation that will be replaced with actual CUDA kernel
__attribute__((visibility("default")))
int relu_kernel(void* data, unsigned int len) {
    return 0; // success
}

} // extern "C"
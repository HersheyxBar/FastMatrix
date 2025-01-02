# FastMatrix

> ⚠️ **Learning Project**: This is my first Rust project as I learn the language! While the code might not follow all Rust best practices, I'm actively learning and improving. Feedback and suggestions are welcome!

FastMatrix is a Rust library designed for high-performance linear algebra operations, specifically targeting machine learning, scientific computing, and image processing tasks.

## Key Features

### Matrix Operations
- Addition, subtraction, and multiplication on CPU
- Element-wise operations (ReLU, sigmoid, tanh)
- Reduction operations (sum, mean, max)

### Neural Network Support
- Basic linear layers with activation functions for quick neural network setup

### Image Processing
- 2D convolution for image manipulation

### Ease of Use
- Simple, clean Rust API
- Example-driven documentation
- Beginner-friendly code with detailed comments explaining the implementations

## Why FastMatrix?

- **Learning Journey**: Follow along as I explore Rust
- **Performance**: Learning to optimize code for speed in computation-intensive tasks
- **Versatility**: Experimenting with everything from simple matrix manipulations to complex neural network layers
- **Extensibility**: Designed with learning in mind - easy to understand

## Planned Features

### GPU Acceleration with CUDA
> **Note**: CUDA support is planned but not yet implemented. This will be a learning experience for both Rust and CUDA programming. If you tried to run CUDA operations, the examples were likely executed on CPU with simulated results.

Once I learn more about CUDA development, I plan to implement:
- CUDA-accelerated matrix operations for significant performance boosts
- Custom CUDA kernels for matrix multiplication, convolutions, etc.
- Advanced optimization techniques including:
  - Memory management with shared memory usage
  - Tiling for GPU kernels

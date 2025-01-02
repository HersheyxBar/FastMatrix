//! # Linear Algebra Library
//!
//! This library provides efficient matrix operations optimized
//! for machine learning and scientific computing.
//!
//! ## Features
//! - Basic matrix operations (add, sub, mul)
//! - Reduction operations (sum, mean, max)
//! - Activation functions (ReLU)
//! - Efficient CPU-based computations using ndarray
//! - Benchmarking suite
//! - Example applications in ML and image processing

pub mod core;
pub mod benchmark;

// Re-export key types and modules for convenience
pub use core::matrix::Matrix;
pub use core::error::CudaError;
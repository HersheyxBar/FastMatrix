//! Benchmarking system to compare different matrix operation implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use crate::core::matrix::Matrix;

fn benchmark_matmul(c: &mut Criterion) {
    // Setup
    let rows = 512;
    let cols = 512;
    let batch = 1;
    let size = rows * cols * batch;
    let host_data_a = vec![1.0f32; size];
    let host_data_b = vec![1.0f32; size];

    // Create matrices
    let gpu_a = Matrix::from_slice(&host_data_a, rows, cols, batch).unwrap();
    let gpu_b = Matrix::from_slice(&host_data_b, cols, rows, batch).unwrap();

    // Benchmark
    c.bench_function("matmul_512x512", |b| {
        b.iter(|| {
            let _res = black_box(gpu_a.matmul(&gpu_b).unwrap());
        });
    });
}

criterion_group!(benches, benchmark_matmul);
criterion_main!(benches);
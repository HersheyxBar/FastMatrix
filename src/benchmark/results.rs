//! Module for collecting and visualizing benchmark results.
//! In practice, you might output JSON or CSV, then use Python/Jupyter for graphs.

#[derive(Debug, serde::Serialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub runtime_ms: f64,
    pub gflops: f64,
}

impl BenchmarkResult {
    pub fn new(name: &str, runtime_ms: f64, gflops: f64) -> Self {
        Self {
            name: name.to_string(),
            runtime_ms,
            gflops,
        }
    }
}

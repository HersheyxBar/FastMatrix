/// Custom error type for CUDA operations and matrix management
#[derive(Debug)]
pub enum CudaError {
    /// Error from host side (e.g., invalid dimensions, memory allocation)
    HostError(String),
    /// Error from CUDA side
    CudaError(String),
    /// Catch-all for other FFI or GPU-related errors
    Other(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::HostError(msg)
            | CudaError::CudaError(msg)
            | CudaError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

use crate::core::error::CudaError;
use libc::c_void;
use std::ffi::c_int;
use std::ptr::null_mut;

/// Manages GPU memory allocations using raw pointers.
/// Typically, you'd use something like `rustacuda`, `cust`, or official CUDA APIs directly.
pub struct CudaAllocator;

impl CudaAllocator {
    /// Allocate GPU memory of `size` bytes.
    pub fn malloc(size: usize) -> Result<*mut c_void, CudaError> {
        unsafe {
            let mut dev_ptr: *mut c_void = null_mut();
            let ret = crate::cuda::wrapper::cuda_malloc_wrapper(&mut dev_ptr, size);
            if ret != 0 {
                return Err(CudaError::CudaError(format!(
                    "cudaMalloc failed with code {}",
                    ret
                )));
            }
            Ok(dev_ptr)
        }
    }

    /// Free GPU memory.
    pub fn free(dev_ptr: *mut c_void) -> Result<(), CudaError> {
        unsafe {
            let ret = crate::cuda::wrapper::cuda_free_wrapper(dev_ptr);
            if ret != 0 {
                return Err(CudaError::CudaError(format!(
                    "cudaFree failed with code {}",
                    ret
                )));
            }
            Ok(())
        }
    }

    /// Copy data from host (CPU) to device (GPU).
    pub fn memcpy_htod(dev_ptr: *mut c_void, host_ptr: *const c_void, size: usize) -> Result<(), CudaError> {
        unsafe {
            let ret = crate::cuda::wrapper::cuda_memcpy_htod_wrapper(dev_ptr, host_ptr, size);
            if ret != 0 {
                return Err(CudaError::CudaError(format!(
                    "cudaMemcpy HtoD failed with code {}",
                    ret
                )));
            }
            Ok(())
        }
    }

    /// Copy data from device (GPU) to host (CPU).
    pub fn memcpy_dtoh(host_ptr: *mut c_void, dev_ptr: *const c_void, size: usize) -> Result<(), CudaError> {
        unsafe {
            let ret = crate::cuda::wrapper::cuda_memcpy_dtoh_wrapper(host_ptr, dev_ptr, size);
            if ret != 0 {
                return Err(CudaError::CudaError(format!(
                    "cudaMemcpy DtoH failed with code {}",
                    ret
                )));
            }
            Ok(())
        }
    }
}

use gpu_accel_linalg::core::matrix::CudaMatrix;
use gpu_accel_linalg::core::error::CudaError;

fn main() -> Result<(), CudaError> {
    env_logger::init();

    // Suppose we have a 16x16 image, single channel, for demonstration.
    let width = 16;
    let height = 16;
    let batch_size = 1;
    let host_image = vec![1.0f32; width * height];

    // A simple 3x3 filter:
    let host_filter = vec![0.1111f32; 3 * 3]; 

    // Transfer to GPU
    let d_image = CudaMatrix::from_slice(&host_image, height, width, batch_size)?;
    let d_filter = CudaMatrix::from_slice(&host_filter, 3, 3, 1)?;

    println!("Image convolution done.");

    // Transfer back to host if desired:
    let _host_result = d_image.to_vec()?;

    Ok(())
}

use gpu_accel_linalg::core::matrix::Matrix;
use gpu_accel_linalg::core::error::CudaError;

fn main() -> Result<(), CudaError> {
    env_logger::init();

    // Create a simple 16x16 image with a pattern
    let width = 16;
    let height = 16;
    let batch_size = 1;

    // Create a simple pattern in the image (a cross)
    let mut host_image = vec![0.0f32; width * height];
    for i in 0..width {
        for j in 0..height {
            if i == height/2 || j == width/2 {
                host_image[i * width + j] = 1.0;
            }
        }
    }

    // A 3x3 edge detection filter
    let host_filter = vec![
        -1.0f32, -1.0, -1.0,
        -1.0, 8.0, -1.0,
        -1.0, -1.0, -1.0
    ];

    // Create matrices
    let image = Matrix::from_slice(&host_image, height, width, batch_size)?;
    let filter = Matrix::from_slice(&host_filter, 3, 3, 1)?;

    // Apply convolution
    let result = image.convolve(&filter)?;

    println!("Image processing completed!");
    println!("Original image dimensions: {}x{}", height, width);
    println!("Processed image dimensions: {}x{}", result.shape().0, result.shape().1);

    // Print the center row of the result to see the edge detection effect
    let result_vec = result.to_vec();
    println!("Center row values: {:?}", 
        &result_vec[width * height/2..width * (height/2 + 1)]);

    Ok(())
}
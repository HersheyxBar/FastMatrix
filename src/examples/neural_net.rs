use gpu_accel_linalg::core::matrix::Matrix;
use gpu_accel_linalg::core::error::CudaError; //Keeping this for potential future use or compatibility

fn main() -> Result<(), String> {
    // Initialize logger
    env_logger::init();

    let input_size = 16;
    let output_size = 8;
    let batch_size = 1;

    // Create random or dummy data
    let host_input = vec![1.0f32; input_size * batch_size];
    let host_weights = vec![0.5f32; input_size * output_size];
    let host_bias = vec![0.1f32; output_size];

    // Create matrices
    let input = Matrix::from_slice(&host_input, 1, input_size, batch_size);
    let weights = Matrix::from_slice(&host_weights, input_size, output_size, 1);
    let bias = Matrix::from_slice(&host_bias, 1, output_size, 1);

    // Linear layer: out = input * weights
    let mut out = input.matmul(&weights)?;

    // Apply ReLU activation
    out.relu();

    println!("Neural network forward pass done.");

    // Get output to inspect
    let out_host = out.to_vec();
    println!("Output: {:?}", &out_host[..]);

    Ok(())
}
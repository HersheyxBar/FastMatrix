fn main() {
    // The cargo-cuda build script will automatically compile the CUDA sources
    // declared in Cargo.toml under [package.metadata.cargo-cuda].
    println!("cargo:rerun-if-changed=src/cuda/kernels/matmul.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/activations.cu");
}
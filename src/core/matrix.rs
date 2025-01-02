use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use crate::core::error::CudaError;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub batch_size: usize,
    data: Array2<T>,
}

impl<T> Matrix<T>
where
    T: Clone + Copy + Default + std::fmt::Debug + Float + 'static,
{
    /// Create a new matrix, uninitialized.
    pub fn new(rows: usize, cols: usize, batch_size: usize) -> Result<Self, CudaError> {
        let data = Array2::default((rows * batch_size, cols));
        Ok(Self {
            rows,
            cols,
            batch_size,
            data,
        })
    }

    /// Create from a slice.
    pub fn from_slice(host_data: &[T], rows: usize, cols: usize, batch_size: usize) -> Result<Self, CudaError> {
        if host_data.len() != rows * cols * batch_size {
            return Err(CudaError::HostError("Shape mismatch in from_slice".to_string()));
        }
        let data = Array2::from_shape_vec((rows * batch_size, cols), host_data.to_vec())
            .map_err(|e| CudaError::HostError(format!("Failed to create array: {}", e)))?;
        Ok(Self {
            rows,
            cols,
            batch_size,
            data,
        })
    }

    /// Get data as a vector
    pub fn to_vec(&self) -> Vec<T> {
        self.data.as_slice().unwrap().to_vec()
    }

    /// Matrix multiplication
    pub fn matmul(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, CudaError> {
        if self.cols != rhs.rows {
            return Err(CudaError::HostError(
                "Dimension mismatch in matmul".to_string(),
            ));
        }

        let result = self.data.dot(&rhs.data);
        Ok(Matrix {
            rows: self.rows,
            cols: rhs.cols,
            batch_size: self.batch_size,
            data: result,
        })
    }

    /// Element-wise addition of bias
    pub fn add_bias(&mut self, bias: &Matrix<T>) -> Result<(), CudaError> {
        if bias.rows != 1 || bias.cols != self.cols {
            return Err(CudaError::HostError(
                "Bias dimensions must match matrix columns".to_string(),
            ));
        }

        for mut row in self.data.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val = *val + bias.data[[0, i]];
            }
        }
        Ok(())
    }

    /// Element-wise ReLU activation
    pub fn relu(&mut self) {
        self.data.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
    }

    /// Basic 2D convolution operation
    pub fn convolve(&self, kernel: &Matrix<T>) -> Result<Matrix<T>, CudaError> {
        if kernel.rows % 2 == 0 || kernel.cols % 2 == 0 {
            return Err(CudaError::HostError(
                "Kernel dimensions must be odd numbers".to_string(),
            ));
        }

        let pad_h = kernel.rows / 2;
        let pad_w = kernel.cols / 2;
        let mut result = Matrix::new(self.rows, self.cols, self.batch_size)?;

        // Simple convolution implementation
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = T::zero();
                for ki in 0..kernel.rows {
                    for kj in 0..kernel.cols {
                        let ii = i as isize + (ki as isize - pad_h as isize);
                        let jj = j as isize + (kj as isize - pad_w as isize);

                        if ii >= 0 && ii < self.rows as isize && jj >= 0 && jj < self.cols as isize {
                            sum = sum + self.data[[ii as usize, jj as usize]] * kernel.data[[ki, kj]];
                        }
                    }
                }
                result.data[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols * self.batch_size
    }

    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get a view of the underlying ndarray
    pub fn as_array(&self) -> ArrayView2<T> {
        self.data.view()
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        // No need for explicit cleanup with ndarray
    }
}
use crate::algorithm::{kmeans_double_chunked, predict_labels};
use crate::config::KMeansConfig;
use crate::error::KMeansError;
use ndarray::{Array1, Array2, ArrayView2};

/// Fast k-means clustering implementation compatible with ndarray.
///
/// This implementation uses double-chunking to process large datasets efficiently
/// without running out of memory. It provides an API similar to FAISS and scikit-learn.
///
/// # Example
///
/// ```
/// use fastkmeans_rs::FastKMeans;
/// use ndarray::Array2;
/// use ndarray_rand::RandomExt;
/// use ndarray_rand::rand_distr::Uniform;
///
/// // Generate random data
/// let data = Array2::random((1000, 128), Uniform::new(-1.0f32, 1.0));
///
/// // Create and train the model
/// let mut kmeans = FastKMeans::new(128, 10);
/// kmeans.train(&data.view()).unwrap();
///
/// // Get cluster assignments
/// let labels = kmeans.predict(&data.view()).unwrap();
/// ```
pub struct FastKMeans {
    /// Model configuration
    config: KMeansConfig,

    /// Number of features (dimensions)
    d: usize,

    /// Trained centroids (None if not yet fitted)
    centroids: Option<Array2<f32>>,
}

impl FastKMeans {
    /// Create a new FastKMeans instance with default configuration.
    ///
    /// # Arguments
    ///
    /// * `d` - Number of features (dimensions) in the data
    /// * `k` - Number of clusters
    ///
    /// # Panics
    ///
    /// Panics if `k` is 0.
    pub fn new(d: usize, k: usize) -> Self {
        assert!(k > 0, "k must be greater than 0");

        Self {
            config: KMeansConfig::new(k),
            d,
            centroids: None,
        }
    }

    /// Create a new FastKMeans instance with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom configuration for the k-means algorithm
    ///
    /// # Panics
    ///
    /// Panics if `config.k` is 0.
    pub fn with_config(config: KMeansConfig) -> Self {
        assert!(config.k > 0, "k must be greater than 0");

        Self {
            d: 0, // Will be set on first train call
            config,
            centroids: None,
        }
    }

    /// Train the k-means model on the given data.
    ///
    /// This method mimics the FAISS `train()` API.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data of shape (n_samples, n_features)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Number of samples is less than k
    /// - Data dimensions don't match (for subsequent calls)
    pub fn train(&mut self, data: &ArrayView2<f32>) -> Result<(), KMeansError> {
        let n_features = data.ncols();

        // Set dimensions on first call, validate on subsequent calls
        if self.d == 0 {
            self.d = n_features;
        } else if n_features != self.d {
            return Err(KMeansError::InvalidDimensions(format!(
                "Expected {} features, got {}",
                self.d, n_features
            )));
        }

        // Run the k-means algorithm
        let result = kmeans_double_chunked(data, &self.config)?;

        self.centroids = Some(result.centroids);
        Ok(())
    }

    /// Fit the model to the data.
    ///
    /// This method mimics the scikit-learn `fit()` API.
    /// It is equivalent to `train()`.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn fit(&mut self, data: &ArrayView2<f32>) -> Result<&mut Self, KMeansError> {
        self.train(data)?;
        Ok(self)
    }

    /// Predict cluster assignments for new data.
    ///
    /// This method mimics the scikit-learn `predict()` API.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to predict, of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Returns an array of cluster labels of shape (n_samples,).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model has not been fitted yet
    /// - Data dimensions don't match the training data
    pub fn predict(&self, data: &ArrayView2<f32>) -> Result<Array1<i64>, KMeansError> {
        let centroids = self.centroids.as_ref().ok_or(KMeansError::NotFitted)?;

        let n_features = data.ncols();
        if n_features != self.d {
            return Err(KMeansError::InvalidDimensions(format!(
                "Expected {} features, got {}",
                self.d, n_features
            )));
        }

        let labels = predict_labels(
            data,
            &centroids.view(),
            self.config.chunk_size_data,
            self.config.chunk_size_centroids,
        );

        Ok(labels)
    }

    /// Fit the model and predict cluster assignments in one call.
    ///
    /// This method mimics the scikit-learn `fit_predict()` API.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Returns an array of cluster labels of shape (n_samples,).
    pub fn fit_predict(&mut self, data: &ArrayView2<f32>) -> Result<Array1<i64>, KMeansError> {
        self.train(data)?;
        self.predict(data)
    }

    /// Get the centroids of the fitted model.
    ///
    /// # Returns
    ///
    /// Returns `Some(&Array2<f32>)` if the model has been fitted, `None` otherwise.
    pub fn centroids(&self) -> Option<&Array2<f32>> {
        self.centroids.as_ref()
    }

    /// Get the number of clusters.
    pub fn k(&self) -> usize {
        self.config.k
    }

    /// Get the number of features (dimensions).
    pub fn d(&self) -> usize {
        self.d
    }

    /// Get the configuration.
    pub fn config(&self) -> &KMeansConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_fastkmeans_new() {
        let kmeans = FastKMeans::new(128, 10);
        assert_eq!(kmeans.k(), 10);
        assert_eq!(kmeans.d(), 128);
        assert!(kmeans.centroids().is_none());
    }

    #[test]
    fn test_fastkmeans_train() {
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeans::new(32, 5);

        kmeans.train(&data.view()).unwrap();

        assert!(kmeans.centroids().is_some());
        let centroids = kmeans.centroids().unwrap();
        assert_eq!(centroids.nrows(), 5);
        assert_eq!(centroids.ncols(), 32);
    }

    #[test]
    fn test_fastkmeans_fit() {
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeans::new(32, 5);

        let result = kmeans.fit(&data.view());
        assert!(result.is_ok());
        assert!(kmeans.centroids().is_some());
    }

    #[test]
    fn test_fastkmeans_predict() {
        let train_data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));
        let test_data = Array2::random((100, 16), Uniform::new(-1.0f32, 1.0));

        let mut kmeans = FastKMeans::new(16, 8);
        kmeans.train(&train_data.view()).unwrap();

        let labels = kmeans.predict(&test_data.view()).unwrap();
        assert_eq!(labels.len(), 100);

        for &label in labels.iter() {
            assert!(label >= 0 && label < 8);
        }
    }

    #[test]
    fn test_fastkmeans_fit_predict() {
        let data = Array2::random((300, 8), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeans::new(8, 4);

        let labels = kmeans.fit_predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 300);
        assert!(kmeans.centroids().is_some());
    }

    #[test]
    fn test_fastkmeans_predict_before_fit() {
        let data = Array2::random((100, 8), Uniform::new(-1.0f32, 1.0));
        let kmeans = FastKMeans::new(8, 5);

        let result = kmeans.predict(&data.view());
        assert!(matches!(result, Err(KMeansError::NotFitted)));
    }

    #[test]
    fn test_fastkmeans_dimension_mismatch() {
        let train_data = Array2::random((100, 8), Uniform::new(-1.0f32, 1.0));
        let test_data = Array2::random((50, 16), Uniform::new(-1.0f32, 1.0));

        let mut kmeans = FastKMeans::new(8, 5);
        kmeans.train(&train_data.view()).unwrap();

        let result = kmeans.predict(&test_data.view());
        assert!(matches!(result, Err(KMeansError::InvalidDimensions(_))));
    }

    #[test]
    #[should_panic(expected = "k must be greater than 0")]
    fn test_fastkmeans_k_zero() {
        let _ = FastKMeans::new(8, 0);
    }
}

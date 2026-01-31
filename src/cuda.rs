//! CUDA-accelerated k-means clustering
//!
//! This module provides GPU-accelerated k-means clustering using CUDA.
//! Enable the `cuda` feature to use this functionality.
//!
//! # Example
//!
//! ```ignore
//! use fastkmeans_rs::cuda::FastKMeansCuda;
//! use fastkmeans_rs::KMeansConfig;
//! use ndarray::Array2;
//! use ndarray_rand::RandomExt;
//! use ndarray_rand::rand_distr::Uniform;
//!
//! let data = Array2::random((10000, 128), Uniform::new(-1.0f32, 1.0));
//!
//! let mut kmeans = FastKMeansCuda::new(128, 50).unwrap();
//! kmeans.train(&data.view()).unwrap();
//!
//! let labels = kmeans.predict(&data.view()).unwrap();
//! ```

use crate::config::KMeansConfig;
use crate::error::KMeansError;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::{Array1, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::time::Instant;

/// CUDA kernels for k-means operations
const CUDA_KERNELS: &str = r#"
extern "C" __global__ void compute_squared_norms(
    const float* data,
    float* norms,
    int n_samples,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        float sum = 0.0f;
        const float* row = data + idx * n_features;
        for (int j = 0; j < n_features; j++) {
            float val = row[j];
            sum += val * val;
        }
        norms[idx] = sum;
    }
}

extern "C" __global__ void find_nearest_centroids(
    const float* data_norms,
    const float* centroid_norms,
    const float* dot_products,
    long long* labels,
    float* best_dists,
    int n_data,
    int n_centroids,
    int centroid_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_data) {
        float x_norm = data_norms[idx];
        float best_dist = best_dists[idx];
        long long best_label = labels[idx];

        for (int j = 0; j < n_centroids; j++) {
            float c_norm = centroid_norms[j];
            float dot = dot_products[idx * n_centroids + j];
            float dist = x_norm + c_norm - 2.0f * dot;

            if (dist < best_dist) {
                best_dist = dist;
                best_label = centroid_offset + j;
            }
        }

        best_dists[idx] = best_dist;
        labels[idx] = best_label;
    }
}

extern "C" __global__ void accumulate_cluster_sums(
    const float* data,
    const long long* labels,
    float* cluster_sums,
    float* cluster_counts,
    int n_samples,
    int n_features,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        int cluster_id = (int)labels[idx];
        if (cluster_id >= 0 && cluster_id < k) {
            atomicAdd(&cluster_counts[cluster_id], 1.0f);
            const float* point = data + idx * n_features;
            float* centroid_sum = cluster_sums + cluster_id * n_features;
            for (int j = 0; j < n_features; j++) {
                atomicAdd(&centroid_sum[j], point[j]);
            }
        }
    }
}

extern "C" __global__ void update_centroids(
    float* centroids,
    const float* cluster_sums,
    const float* cluster_counts,
    int k,
    int n_features
) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id < k) {
        float count = cluster_counts[cluster_id];
        if (count > 0.0f) {
            float* centroid = centroids + cluster_id * n_features;
            const float* sum = cluster_sums + cluster_id * n_features;
            for (int j = 0; j < n_features; j++) {
                centroid[j] = sum[j] / count;
            }
        }
    }
}

extern "C" __global__ void compute_centroid_shift(
    const float* old_centroids,
    const float* new_centroids,
    float* shifts,
    int k,
    int n_features
) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id < k) {
        const float* old_c = old_centroids + cluster_id * n_features;
        const float* new_c = new_centroids + cluster_id * n_features;
        float diff_sq = 0.0f;
        for (int j = 0; j < n_features; j++) {
            float d = new_c[j] - old_c[j];
            diff_sq += d * d;
        }
        shifts[cluster_id] = sqrtf(diff_sq);
    }
}

extern "C" __global__ void init_best_dists(
    float* best_dists,
    int n_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        best_dists[idx] = 3.4028235e+38f;  // FLT_MAX
    }
}

extern "C" __global__ void init_labels(
    long long* labels,
    int n_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        labels[idx] = 0;
    }
}
"#;

const MODULE_NAME: &str = "kmeans_kernels";

/// CUDA-accelerated k-means clustering
pub struct FastKMeansCuda {
    /// Model configuration
    config: KMeansConfig,

    /// Number of features (dimensions)
    d: usize,

    /// Trained centroids (None if not yet fitted)
    centroids: Option<Array2<f32>>,

    /// CUDA device
    device: Arc<CudaDevice>,

    /// cuBLAS handle
    blas: CudaBlas,
}

impl FastKMeansCuda {
    /// Create a new FastKMeansCuda instance with default configuration.
    ///
    /// # Arguments
    ///
    /// * `d` - Number of features (dimensions) in the data
    /// * `k` - Number of clusters
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization fails.
    pub fn new(d: usize, k: usize) -> Result<Self, KMeansError> {
        assert!(k > 0, "k must be greater than 0");
        Self::with_config_and_device(KMeansConfig::new(k), Some(d), 0)
    }

    /// Create a new FastKMeansCuda instance with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom configuration for the k-means algorithm
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization fails.
    pub fn with_config(config: KMeansConfig) -> Result<Self, KMeansError> {
        assert!(config.k > 0, "k must be greater than 0");
        Self::with_config_and_device(config, None, 0)
    }

    /// Create a new FastKMeansCuda instance with a specific GPU device.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom configuration
    /// * `d` - Optional number of features
    /// * `device_id` - CUDA device ID (0, 1, 2, ...)
    pub fn with_config_and_device(
        config: KMeansConfig,
        d: Option<usize>,
        device_id: usize,
    ) -> Result<Self, KMeansError> {
        // Initialize CUDA device
        let device = CudaDevice::new(device_id).map_err(|e| {
            KMeansError::InvalidK(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        // Compile PTX
        let ptx = compile_ptx(CUDA_KERNELS)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to compile CUDA kernels: {}", e)))?;

        // Load module with all functions
        device
            .load_ptx(
                ptx,
                MODULE_NAME,
                &[
                    "compute_squared_norms",
                    "find_nearest_centroids",
                    "accumulate_cluster_sums",
                    "update_centroids",
                    "compute_centroid_shift",
                    "init_best_dists",
                    "init_labels",
                ],
            )
            .map_err(|e| KMeansError::InvalidK(format!("Failed to load CUDA module: {}", e)))?;

        // Create cuBLAS handle
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| KMeansError::InvalidK(format!("Failed to create cuBLAS handle: {}", e)))?;

        Ok(Self {
            config,
            d: d.unwrap_or(0),
            centroids: None,
            device,
            blas,
        })
    }

    /// Get a CUDA function by name
    fn get_func(&self, name: &str) -> Result<CudaFunction, KMeansError> {
        self.device
            .get_func(MODULE_NAME, name)
            .ok_or_else(|| KMeansError::InvalidK(format!("Failed to get CUDA function: {}", name)))
    }

    /// Train the k-means model on the given data using GPU acceleration.
    pub fn train(&mut self, data: &ArrayView2<f32>) -> Result<(), KMeansError> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = self.config.k;

        // Set dimensions on first call
        if self.d == 0 {
            self.d = n_features;
        } else if n_features != self.d {
            return Err(KMeansError::InvalidDimensions(format!(
                "Expected {} features, got {}",
                self.d, n_features
            )));
        }

        // Validate inputs
        if k == 0 {
            return Err(KMeansError::InvalidK(
                "k must be greater than 0".to_string(),
            ));
        }

        if n_samples < k {
            return Err(KMeansError::InsufficientData(format!(
                "Number of samples ({}) is less than k ({})",
                n_samples, k
            )));
        }

        // Initialize RNG
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);

        // Subsample if needed
        let (data_subset, _) = self.subsample_data(data, &mut rng)?;
        let n_samples_used = data_subset.nrows();

        if self.config.verbose {
            eprintln!(
                "[CUDA] Training k-means: {} samples ({}), {} features, {} clusters",
                n_samples_used,
                if n_samples_used < n_samples {
                    format!("subsampled from {}", n_samples)
                } else {
                    "full data".to_string()
                },
                n_features,
                k
            );
        }

        // Copy data to GPU (row-major layout)
        let data_flat: Vec<f32> = data_subset.as_standard_layout().iter().cloned().collect();
        let d_data: CudaSlice<f32> = self
            .device
            .htod_sync_copy(&data_flat)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to copy data to GPU: {}", e)))?;

        // Compute data norms on GPU
        let d_data_norms = self.compute_squared_norms_gpu(&d_data, n_samples_used, n_features)?;

        // Initialize centroids (random selection from data)
        let mut centroids = self.initialize_centroids(&data_subset.view(), k, &mut rng);

        // Allocate GPU buffers for the main loop
        let mut d_labels: CudaSlice<i64> = self
            .device
            .alloc_zeros(n_samples_used)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate labels: {}", e)))?;
        let mut d_best_dists: CudaSlice<f32> = self
            .device
            .alloc_zeros(n_samples_used)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate best_dists: {}", e)))?;

        // Main k-means loop
        for iteration in 0..self.config.max_iters {
            let iter_start = Instant::now();

            // Copy centroids to GPU
            let centroids_flat: Vec<f32> = centroids.as_standard_layout().iter().cloned().collect();
            let d_centroids: CudaSlice<f32> =
                self.device.htod_sync_copy(&centroids_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy centroids to GPU: {}", e))
                })?;

            // Compute centroid norms
            let d_centroid_norms = self.compute_squared_norms_gpu(&d_centroids, k, n_features)?;

            // Initialize best distances to infinity
            self.init_best_dists_gpu(&mut d_best_dists, n_samples_used)?;

            // Find nearest centroids using chunking for memory efficiency
            self.find_nearest_centroids_gpu(
                &d_data,
                &d_data_norms,
                &d_centroids,
                &d_centroid_norms,
                &mut d_labels,
                &mut d_best_dists,
                n_samples_used,
                n_features,
                k,
            )?;

            // Accumulate cluster sums and counts
            let (cluster_sums, cluster_counts) =
                self.accumulate_clusters_gpu(&d_data, &d_labels, n_samples_used, n_features, k)?;

            // Save old centroids for convergence check
            let prev_centroids = centroids.clone();

            // Update centroids
            let mut empty_clusters = Vec::new();
            for cluster_idx in 0..k {
                let count = cluster_counts[cluster_idx];
                if count > 0.0 {
                    for j in 0..n_features {
                        centroids[[cluster_idx, j]] = cluster_sums[[cluster_idx, j]] / count;
                    }
                } else {
                    empty_clusters.push(cluster_idx);
                }
            }

            // Reinitialize empty clusters
            if !empty_clusters.is_empty() {
                let indices: Vec<usize> = (0..n_samples_used).collect();
                let random_indices: Vec<usize> = indices
                    .choose_multiple(&mut rng, empty_clusters.len())
                    .cloned()
                    .collect();

                for (i, &cluster_idx) in empty_clusters.iter().enumerate() {
                    let data_idx = random_indices[i];
                    centroids
                        .row_mut(cluster_idx)
                        .assign(&data_subset.row(data_idx));
                }

                if self.config.verbose {
                    eprintln!(
                        "  [CUDA] Reinitialized {} empty clusters",
                        empty_clusters.len()
                    );
                }
            }

            // Check convergence
            let shift = self.compute_shift(&prev_centroids.view(), &centroids.view());

            if self.config.verbose {
                let iter_time = iter_start.elapsed().as_secs_f64();
                eprintln!(
                    "  [CUDA] Iteration {}/{}: shift = {:.6}, time = {:.4}s",
                    iteration + 1,
                    self.config.max_iters,
                    shift,
                    iter_time
                );
            }

            if self.config.tol >= 0.0 && shift < self.config.tol {
                if self.config.verbose {
                    eprintln!(
                        "  [CUDA] Converged after {} iterations (shift {:.6} < tol {:.6})",
                        iteration + 1,
                        shift,
                        self.config.tol
                    );
                }
                break;
            }
        }

        self.centroids = Some(centroids);
        Ok(())
    }

    /// Fit the model to the data (scikit-learn style API).
    pub fn fit(&mut self, data: &ArrayView2<f32>) -> Result<&mut Self, KMeansError> {
        self.train(data)?;
        Ok(self)
    }

    /// Predict cluster assignments for new data using GPU acceleration.
    pub fn predict(&self, data: &ArrayView2<f32>) -> Result<Array1<i64>, KMeansError> {
        let centroids = self.centroids.as_ref().ok_or(KMeansError::NotFitted)?;
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = centroids.nrows();

        if n_features != self.d {
            return Err(KMeansError::InvalidDimensions(format!(
                "Expected {} features, got {}",
                self.d, n_features
            )));
        }

        // Copy data to GPU
        let data_flat: Vec<f32> = data.as_standard_layout().iter().cloned().collect();
        let d_data: CudaSlice<f32> = self
            .device
            .htod_sync_copy(&data_flat)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to copy data to GPU: {}", e)))?;

        // Copy centroids to GPU
        let centroids_flat: Vec<f32> = centroids.as_standard_layout().iter().cloned().collect();
        let d_centroids: CudaSlice<f32> =
            self.device.htod_sync_copy(&centroids_flat).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to copy centroids to GPU: {}", e))
            })?;

        // Compute norms
        let d_data_norms = self.compute_squared_norms_gpu(&d_data, n_samples, n_features)?;
        let d_centroid_norms = self.compute_squared_norms_gpu(&d_centroids, k, n_features)?;

        // Allocate output buffers
        let mut d_labels: CudaSlice<i64> = self
            .device
            .alloc_zeros(n_samples)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate labels: {}", e)))?;
        let mut d_best_dists: CudaSlice<f32> = self
            .device
            .alloc_zeros(n_samples)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate best_dists: {}", e)))?;

        // Initialize
        self.init_best_dists_gpu(&mut d_best_dists, n_samples)?;

        // Find nearest centroids
        self.find_nearest_centroids_gpu(
            &d_data,
            &d_data_norms,
            &d_centroids,
            &d_centroid_norms,
            &mut d_labels,
            &mut d_best_dists,
            n_samples,
            n_features,
            k,
        )?;

        // Copy results back
        let labels_vec = self
            .device
            .dtoh_sync_copy(&d_labels)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to copy labels from GPU: {}", e)))?;

        Ok(Array1::from_vec(labels_vec))
    }

    /// Fit the model and predict cluster assignments in one call.
    pub fn fit_predict(&mut self, data: &ArrayView2<f32>) -> Result<Array1<i64>, KMeansError> {
        self.train(data)?;
        self.predict(data)
    }

    /// Get the centroids of the fitted model.
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

    // =========================================================================
    // Private helper methods
    // =========================================================================

    fn compute_squared_norms_gpu(
        &self,
        d_data: &CudaSlice<f32>,
        n_samples: usize,
        n_features: usize,
    ) -> Result<CudaSlice<f32>, KMeansError> {
        let mut d_norms: CudaSlice<f32> = self
            .device
            .alloc_zeros(n_samples)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate norms: {}", e)))?;

        let block_size = 256;
        let grid_size = (n_samples + block_size - 1) / block_size;

        let func = self.get_func("compute_squared_norms")?;

        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (d_data, &mut d_norms, n_samples as i32, n_features as i32),
            )
        }
        .map_err(|e| KMeansError::InvalidK(format!("Failed to launch kernel: {}", e)))?;

        Ok(d_norms)
    }

    fn init_best_dists_gpu(
        &self,
        d_best_dists: &mut CudaSlice<f32>,
        n_samples: usize,
    ) -> Result<(), KMeansError> {
        let block_size = 256;
        let grid_size = (n_samples + block_size - 1) / block_size;

        let func = self.get_func("init_best_dists")?;

        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, (d_best_dists, n_samples as i32)) }
            .map_err(|e| KMeansError::InvalidK(format!("Failed to launch kernel: {}", e)))?;

        Ok(())
    }

    fn find_nearest_centroids_gpu(
        &self,
        d_data: &CudaSlice<f32>,
        d_data_norms: &CudaSlice<f32>,
        d_centroids: &CudaSlice<f32>,
        d_centroid_norms: &CudaSlice<f32>,
        d_labels: &mut CudaSlice<i64>,
        d_best_dists: &mut CudaSlice<f32>,
        n_samples: usize,
        n_features: usize,
        k: usize,
    ) -> Result<(), KMeansError> {
        // Process centroids in chunks to minimize memory for dot products
        let chunk_size = self.config.chunk_size_centroids.min(k);
        let mut c_start = 0;

        while c_start < k {
            let c_end = (c_start + chunk_size).min(k);
            let n_centroids_chunk = c_end - c_start;

            // Allocate dot products matrix (n_samples x n_centroids_chunk)
            // Note: cuBLAS uses column-major, so we'll get the result in column-major format
            let mut d_dot_products: CudaSlice<f32> = self
                .device
                .alloc_zeros(n_samples * n_centroids_chunk)
                .map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to allocate dot products: {}", e))
                })?;

            // Copy centroid chunk to device
            let centroids_all: Vec<f32> = self
                .device
                .dtoh_sync_copy(d_centroids)
                .map_err(|e| KMeansError::InvalidK(format!("Failed to copy centroids: {}", e)))?;
            let centroids_chunk_start = c_start * n_features;
            let centroids_chunk_end = c_end * n_features;
            let d_centroids_chunk: CudaSlice<f32> = self
                .device
                .htod_sync_copy(&centroids_all[centroids_chunk_start..centroids_chunk_end])
                .map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy centroid chunk: {}", e))
                })?;

            // Compute dot products using cuBLAS GEMM
            // We want: result[i,j] = data[i,:] . centroids[j,:]
            // In row-major terms: result = data @ centroids.T
            //
            // cuBLAS uses column-major. For row-major data D (n_samples x n_features) and
            // centroids C (n_centroids x n_features), we want D @ C.T
            //
            // In column-major view:
            // - D is seen as D^T (n_features x n_samples)
            // - C is seen as C^T (n_features x n_centroids)
            //
            // We want (D @ C.T)^T in column-major = C @ D^T
            // So: gemm(C_col, D_col) with C_col = C^T in col-major, D_col = D^T in col-major
            // Result will be (n_centroids x n_samples) in column-major = (n_samples x n_centroids) in row-major
            //
            // Using gemm: C = alpha * op(A) * op(B) + beta * C
            // We want: C_col (result in col-major, shape n_centroids x n_samples)
            //   = centroids_col^T @ data_col
            //   = op(centroids) * op(data) where:
            //     - op(centroids) = T, input shape (n_features x n_centroids), output (n_centroids x n_features)
            //     - op(data) = N, input shape (n_features x n_samples), output (n_features x n_samples)
            //
            // So: m = n_centroids, n = n_samples, k = n_features
            //     transa = T, transb = N
            //     A = centroids, lda = n_features
            //     B = data, ldb = n_features
            //     C = result, ldc = n_centroids

            let gemm_cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_centroids_chunk as i32,
                n: n_samples as i32,
                k: n_features as i32,
                alpha: 1.0f32,
                lda: n_features as i32,
                ldb: n_features as i32,
                beta: 0.0f32,
                ldc: n_centroids_chunk as i32,
            };

            unsafe {
                self.blas
                    .gemm(gemm_cfg, &d_centroids_chunk, d_data, &mut d_dot_products)
            }
            .map_err(|e| KMeansError::InvalidK(format!("cuBLAS GEMM failed: {}", e)))?;

            // Extract centroid norms for this chunk
            let centroid_norms_all: Vec<f32> =
                self.device.dtoh_sync_copy(d_centroid_norms).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy centroid norms: {}", e))
                })?;

            let d_centroid_norms_chunk: CudaSlice<f32> = self
                .device
                .htod_sync_copy(&centroid_norms_all[c_start..c_end])
                .map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy centroid norms chunk: {}", e))
                })?;

            // Update labels and best distances
            let block_size = 256;
            let grid_size = (n_samples + block_size - 1) / block_size;

            let func = self.get_func("find_nearest_centroids")?;

            let cfg = LaunchConfig {
                block_dim: (block_size as u32, 1, 1),
                grid_dim: (grid_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        d_data_norms,
                        &d_centroid_norms_chunk,
                        &d_dot_products,
                        &mut *d_labels,
                        &mut *d_best_dists,
                        n_samples as i32,
                        n_centroids_chunk as i32,
                        c_start as i32,
                    ),
                )
            }
            .map_err(|e| KMeansError::InvalidK(format!("Failed to launch kernel: {}", e)))?;

            c_start = c_end;
        }

        Ok(())
    }

    fn accumulate_clusters_gpu(
        &self,
        d_data: &CudaSlice<f32>,
        d_labels: &CudaSlice<i64>,
        n_samples: usize,
        n_features: usize,
        k: usize,
    ) -> Result<(Array2<f32>, Array1<f32>), KMeansError> {
        // Allocate accumulators
        let mut d_cluster_sums: CudaSlice<f32> =
            self.device.alloc_zeros(k * n_features).map_err(|e| {
                KMeansError::InvalidK(format!("Failed to allocate cluster sums: {}", e))
            })?;
        let mut d_cluster_counts: CudaSlice<f32> = self.device.alloc_zeros(k).map_err(|e| {
            KMeansError::InvalidK(format!("Failed to allocate cluster counts: {}", e))
        })?;

        // Launch accumulation kernel
        let block_size = 256;
        let grid_size = (n_samples + block_size - 1) / block_size;

        let func = self.get_func("accumulate_cluster_sums")?;

        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    d_data,
                    d_labels,
                    &mut d_cluster_sums,
                    &mut d_cluster_counts,
                    n_samples as i32,
                    n_features as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| KMeansError::InvalidK(format!("Failed to launch kernel: {}", e)))?;

        // Copy results back to host
        let sums_vec = self
            .device
            .dtoh_sync_copy(&d_cluster_sums)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to copy cluster sums: {}", e)))?;
        let counts_vec = self
            .device
            .dtoh_sync_copy(&d_cluster_counts)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to copy cluster counts: {}", e)))?;

        let cluster_sums = Array2::from_shape_vec((k, n_features), sums_vec)
            .map_err(|e| KMeansError::InvalidK(format!("Failed to reshape sums: {}", e)))?;
        let cluster_counts = Array1::from_vec(counts_vec);

        Ok((cluster_sums, cluster_counts))
    }

    fn subsample_data(
        &self,
        data: &ArrayView2<f32>,
        rng: &mut ChaCha8Rng,
    ) -> Result<(Array2<f32>, Option<Vec<usize>>), KMeansError> {
        let n_samples = data.nrows();

        if let Some(max_ppc) = self.config.max_points_per_centroid {
            let max_samples = self.config.k * max_ppc;
            if n_samples > max_samples {
                if self.config.verbose {
                    eprintln!(
                        "[CUDA] Subsampling data from {} to {} samples",
                        n_samples, max_samples
                    );
                }

                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                indices.truncate(max_samples);
                indices.sort_unstable();

                let n_features = data.ncols();
                let mut subset = Array2::zeros((max_samples, n_features));
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    subset.row_mut(new_idx).assign(&data.row(old_idx));
                }

                return Ok((subset, Some(indices)));
            }
        }

        Ok((data.to_owned(), None))
    }

    fn initialize_centroids(
        &self,
        data: &ArrayView2<f32>,
        k: usize,
        rng: &mut ChaCha8Rng,
    ) -> Array2<f32> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        let indices: Vec<usize> = (0..n_samples).collect();
        let selected: Vec<usize> = indices.choose_multiple(rng, k).cloned().collect();

        let mut centroids = Array2::zeros((k, n_features));
        for (centroid_idx, &data_idx) in selected.iter().enumerate() {
            centroids.row_mut(centroid_idx).assign(&data.row(data_idx));
        }

        centroids
    }

    fn compute_shift(
        &self,
        old_centroids: &ArrayView2<f32>,
        new_centroids: &ArrayView2<f32>,
    ) -> f64 {
        let k = old_centroids.nrows();
        let mut total_shift = 0.0f64;

        for i in 0..k {
            let old_c = old_centroids.row(i);
            let new_c = new_centroids.row(i);
            let mut diff_sq = 0.0f64;
            for j in 0..old_c.len() {
                let d = (new_c[j] - old_c[j]) as f64;
                diff_sq += d * d;
            }
            total_shift += diff_sq.sqrt();
        }

        total_shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_cuda_basic_train() {
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansCuda::new(32, 5).unwrap();

        let result = kmeans.train(&data.view());
        assert!(
            result.is_ok(),
            "CUDA training should succeed: {:?}",
            result.err()
        );
        assert!(kmeans.centroids().is_some());

        let centroids = kmeans.centroids().unwrap();
        assert_eq!(centroids.nrows(), 5);
        assert_eq!(centroids.ncols(), 32);
    }

    #[test]
    fn test_cuda_basic_predict() {
        let data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansCuda::new(16, 8).unwrap();

        kmeans.train(&data.view()).unwrap();
        let labels = kmeans.predict(&data.view()).unwrap();

        assert_eq!(labels.len(), 500);
        for &label in labels.iter() {
            assert!((0..8).contains(&label));
        }
    }

    #[test]
    fn test_cuda_fit_predict() {
        let data = Array2::random((300, 8), Uniform::new(-1.0f32, 1.0));
        let mut kmeans = FastKMeansCuda::new(8, 4).unwrap();

        let labels = kmeans.fit_predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 300);
        assert!(kmeans.centroids().is_some());
    }

    #[test]
    fn test_cuda_reproducibility() {
        let data = Array2::random((500, 32), Uniform::new(-1.0f32, 1.0));

        let config1 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);
        let config2 = KMeansConfig::new(5).with_seed(12345).with_max_iters(10);

        let mut kmeans1 = FastKMeansCuda::with_config(config1).unwrap();
        let mut kmeans2 = FastKMeansCuda::with_config(config2).unwrap();

        kmeans1.train(&data.view()).unwrap();
        kmeans2.train(&data.view()).unwrap();

        let centroids1 = kmeans1.centroids().unwrap();
        let centroids2 = kmeans2.centroids().unwrap();

        for i in 0..centroids1.nrows() {
            for j in 0..centroids1.ncols() {
                assert!(
                    (centroids1[[i, j]] - centroids2[[i, j]]).abs() < 1e-4,
                    "CUDA results should be reproducible with same seed"
                );
            }
        }
    }

    #[test]
    fn test_cuda_matches_cpu() {
        let data = Array2::random((200, 16), Uniform::new(-1.0f32, 1.0));

        // CUDA version
        let cuda_config = KMeansConfig::new(5)
            .with_seed(42)
            .with_max_iters(20)
            .with_max_points_per_centroid(None);
        let mut cuda_kmeans = FastKMeansCuda::with_config(cuda_config).unwrap();
        cuda_kmeans.train(&data.view()).unwrap();
        let cuda_labels = cuda_kmeans.predict(&data.view()).unwrap();

        // CPU version
        let cpu_config = KMeansConfig::new(5)
            .with_seed(42)
            .with_max_iters(20)
            .with_max_points_per_centroid(None);
        let mut cpu_kmeans = crate::FastKMeans::with_config(cpu_config);
        cpu_kmeans.train(&data.view()).unwrap();
        let cpu_labels = cpu_kmeans.predict(&data.view()).unwrap();

        // Compare results - they should be identical with same seed
        let cuda_centroids = cuda_kmeans.centroids().unwrap();
        let cpu_centroids = cpu_kmeans.centroids().unwrap();

        // Check that centroids are close (allowing for minor floating point differences)
        let mut max_diff = 0.0f32;
        for i in 0..cuda_centroids.nrows() {
            for j in 0..cuda_centroids.ncols() {
                let diff = (cuda_centroids[[i, j]] - cpu_centroids[[i, j]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        // Allow for some floating point difference due to different computation order
        assert!(
            max_diff < 0.1,
            "CUDA and CPU centroids should be similar (max diff: {})",
            max_diff
        );

        // Labels should mostly match
        let mut matching = 0;
        for i in 0..cuda_labels.len() {
            if cuda_labels[i] == cpu_labels[i] {
                matching += 1;
            }
        }
        let match_ratio = matching as f64 / cuda_labels.len() as f64;
        assert!(
            match_ratio > 0.9,
            "CUDA and CPU labels should mostly match (ratio: {})",
            match_ratio
        );
    }
}

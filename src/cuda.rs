//! CUDA-accelerated k-means clustering with low memory usage
//!
//! This module provides GPU-accelerated k-means clustering using CUDA.
//! It uses double-chunking (chunking both data and centroids) to minimize
//! VRAM and RAM usage while maintaining high performance.
//!
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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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

extern "C" __global__ void zero_float_array(
    float* arr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = 0.0f;
    }
}
"#;

const MODULE_NAME: &str = "kmeans_kernels";

/// CUDA-accelerated k-means clustering with low memory footprint
///
/// This implementation uses double-chunking to process both data and centroids
/// in manageable chunks, minimizing VRAM and RAM usage while maintaining
/// high performance through GPU acceleration.
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
                    "init_best_dists",
                    "init_labels",
                    "zero_float_array",
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
    ///
    /// Uses double-chunking to minimize memory usage: data is processed in
    /// chunks on the host side, and centroids are processed in chunks on GPU.
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
            eprintln!(
                "[CUDA] Using chunk sizes: data={}, centroids={}",
                self.config.chunk_size_data, self.config.chunk_size_centroids
            );
        }

        // Pre-compute data norms on CPU (will be uploaded in chunks)
        let data_norms = self.compute_squared_norms_cpu(&data_subset.view());

        // Initialize centroids (random selection from data)
        let mut centroids = self.initialize_centroids(&data_subset.view(), k, &mut rng);

        // Host-side arrays for labels and accumulation
        let mut labels = Array1::<i64>::zeros(n_samples_used);

        // Main k-means loop
        for iteration in 0..self.config.max_iters {
            let iter_start = Instant::now();

            // Pre-compute centroid norms on CPU
            let centroid_norms = self.compute_squared_norms_cpu(&centroids.view());

            // Process data in chunks (double-chunking: data chunks on host, centroid chunks on GPU)
            let chunk_size_data = self.config.chunk_size_data;
            let mut data_start = 0;

            while data_start < n_samples_used {
                let data_end = (data_start + chunk_size_data).min(n_samples_used);
                let chunk_size = data_end - data_start;

                // Extract data chunk and upload to GPU
                let data_chunk = data_subset.slice(ndarray::s![data_start..data_end, ..]);
                let data_chunk_flat: Vec<f32> = data_chunk.as_standard_layout().iter().cloned().collect();
                let d_data_chunk: CudaSlice<f32> =
                    self.device.htod_sync_copy(&data_chunk_flat).map_err(|e| {
                        KMeansError::InvalidK(format!("Failed to copy data chunk to GPU: {}", e))
                    })?;

                // Upload data norms for this chunk
                let data_norms_chunk: Vec<f32> = data_norms.slice(ndarray::s![data_start..data_end]).to_vec();
                let d_data_norms_chunk: CudaSlice<f32> =
                    self.device.htod_sync_copy(&data_norms_chunk).map_err(|e| {
                        KMeansError::InvalidK(format!("Failed to copy data norms to GPU: {}", e))
                    })?;

                // Allocate labels and best_dists for this chunk
                let mut d_labels_chunk: CudaSlice<i64> = self
                    .device
                    .alloc_zeros(chunk_size)
                    .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate labels: {}", e)))?;
                let mut d_best_dists_chunk: CudaSlice<f32> = self
                    .device
                    .alloc_zeros(chunk_size)
                    .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate best_dists: {}", e)))?;

                // Initialize best distances to infinity
                self.init_best_dists_gpu(&mut d_best_dists_chunk, chunk_size)?;

                // Find nearest centroids using centroid chunking on GPU
                self.find_nearest_centroids_chunked_gpu(
                    &d_data_chunk,
                    &d_data_norms_chunk,
                    &centroids,
                    &centroid_norms,
                    &mut d_labels_chunk,
                    &mut d_best_dists_chunk,
                    chunk_size,
                    n_features,
                    k,
                )?;

                // Copy labels back to host
                let labels_chunk = self
                    .device
                    .dtoh_sync_copy(&d_labels_chunk)
                    .map_err(|e| KMeansError::InvalidK(format!("Failed to copy labels: {}", e)))?;

                // Store labels
                for (i, &label) in labels_chunk.iter().enumerate() {
                    labels[data_start + i] = label;
                }

                // GPU memory for this chunk is automatically freed when d_data_chunk etc. go out of scope
                data_start = data_end;
            }

            // Accumulate cluster sums and counts on CPU (memory efficient)
            let (cluster_sums, cluster_counts) =
                self.accumulate_clusters_cpu(&data_subset.view(), &labels.view(), k);

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
    ///
    /// Uses chunked processing to minimize memory usage.
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

        // Pre-compute centroid norms on CPU
        let centroid_norms = self.compute_squared_norms_cpu(&centroids.view());

        // Process data in chunks
        let mut labels = Array1::<i64>::zeros(n_samples);
        let chunk_size_data = self.config.chunk_size_data;
        let mut data_start = 0;

        while data_start < n_samples {
            let data_end = (data_start + chunk_size_data).min(n_samples);
            let chunk_size = data_end - data_start;

            // Extract and upload data chunk
            let data_chunk = data.slice(ndarray::s![data_start..data_end, ..]);
            let data_chunk_flat: Vec<f32> = data_chunk.as_standard_layout().iter().cloned().collect();
            let d_data_chunk: CudaSlice<f32> =
                self.device.htod_sync_copy(&data_chunk_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy data chunk to GPU: {}", e))
                })?;

            // Compute data norms for this chunk on GPU
            let d_data_norms_chunk = self.compute_squared_norms_gpu(&d_data_chunk, chunk_size, n_features)?;

            // Download data norms (needed for find_nearest_centroids_chunked_gpu which takes host centroid data)
            let data_norms_chunk: Vec<f32> = self
                .device
                .dtoh_sync_copy(&d_data_norms_chunk)
                .map_err(|e| KMeansError::InvalidK(format!("Failed to copy norms: {}", e)))?;
            let d_data_norms_chunk: CudaSlice<f32> =
                self.device.htod_sync_copy(&data_norms_chunk).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy data norms to GPU: {}", e))
                })?;

            // Allocate output buffers for this chunk
            let mut d_labels_chunk: CudaSlice<i64> = self
                .device
                .alloc_zeros(chunk_size)
                .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate labels: {}", e)))?;
            let mut d_best_dists_chunk: CudaSlice<f32> = self
                .device
                .alloc_zeros(chunk_size)
                .map_err(|e| KMeansError::InvalidK(format!("Failed to allocate best_dists: {}", e)))?;

            // Initialize
            self.init_best_dists_gpu(&mut d_best_dists_chunk, chunk_size)?;

            // Find nearest centroids using centroid chunking
            self.find_nearest_centroids_chunked_gpu(
                &d_data_chunk,
                &d_data_norms_chunk,
                centroids,
                &centroid_norms,
                &mut d_labels_chunk,
                &mut d_best_dists_chunk,
                chunk_size,
                n_features,
                k,
            )?;

            // Copy results back
            let labels_chunk = self
                .device
                .dtoh_sync_copy(&d_labels_chunk)
                .map_err(|e| KMeansError::InvalidK(format!("Failed to copy labels: {}", e)))?;

            for (i, &label) in labels_chunk.iter().enumerate() {
                labels[data_start + i] = label;
            }

            data_start = data_end;
        }

        Ok(labels)
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

    /// Compute squared norms on CPU (more memory efficient for large datasets)
    fn compute_squared_norms_cpu(&self, data: &ArrayView2<f32>) -> Array1<f32> {
        let n_samples = data.nrows();
        let mut norms = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = data.row(i);
            norms[i] = row.dot(&row);
        }

        norms
    }

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

    /// Find nearest centroids using centroid chunking on GPU
    ///
    /// This method takes host-side centroids and centroid norms, and uploads
    /// them in chunks to minimize GPU memory usage for the dot product computation.
    fn find_nearest_centroids_chunked_gpu(
        &self,
        d_data: &CudaSlice<f32>,
        d_data_norms: &CudaSlice<f32>,
        centroids: &Array2<f32>,
        centroid_norms: &Array1<f32>,
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

            // Extract centroid chunk from host array and upload
            let centroid_chunk = centroids.slice(ndarray::s![c_start..c_end, ..]);
            let centroid_chunk_flat: Vec<f32> = centroid_chunk.as_standard_layout().iter().cloned().collect();
            let d_centroids_chunk: CudaSlice<f32> =
                self.device.htod_sync_copy(&centroid_chunk_flat).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy centroid chunk: {}", e))
                })?;

            // Extract centroid norms chunk and upload
            let centroid_norms_chunk: Vec<f32> = centroid_norms.slice(ndarray::s![c_start..c_end]).to_vec();
            let d_centroid_norms_chunk: CudaSlice<f32> =
                self.device.htod_sync_copy(&centroid_norms_chunk).map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to copy centroid norms chunk: {}", e))
                })?;

            // Allocate dot products matrix (n_samples x n_centroids_chunk)
            let mut d_dot_products: CudaSlice<f32> = self
                .device
                .alloc_zeros(n_samples * n_centroids_chunk)
                .map_err(|e| {
                    KMeansError::InvalidK(format!("Failed to allocate dot products: {}", e))
                })?;

            // Compute dot products using cuBLAS GEMM
            // We want: result[i,j] = data[i,:] . centroids[j,:]
            // In row-major terms: result = data @ centroids.T
            //
            // cuBLAS uses column-major. For row-major data D (n_samples x n_features) and
            // centroids C (n_centroids x n_features), we want D @ C.T
            //
            // Using gemm: C = alpha * op(A) * op(B) + beta * C
            // Result will be (n_centroids x n_samples) in column-major = (n_samples x n_centroids) in row-major
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

            // GPU memory for centroid chunk and dot products freed when they go out of scope
            c_start = c_end;
        }

        Ok(())
    }

    /// Accumulate cluster sums and counts on CPU
    /// This avoids keeping large intermediate GPU buffers
    fn accumulate_clusters_cpu(
        &self,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<i64>,
        k: usize,
    ) -> (Array2<f32>, Array1<f32>) {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        let mut cluster_sums = Array2::<f32>::zeros((k, n_features));
        let mut cluster_counts = Array1::<f32>::zeros(k);

        for i in 0..n_samples {
            let cluster_idx = labels[i] as usize;
            if cluster_idx < k {
                cluster_counts[cluster_idx] += 1.0;
                for j in 0..n_features {
                    cluster_sums[[cluster_idx, j]] += data[[i, j]];
                }
            }
        }

        (cluster_sums, cluster_counts)
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

    #[test]
    fn test_cuda_chunked_processing() {
        // Test with small chunk sizes to verify chunking works correctly
        let data = Array2::random((1000, 32), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig::new(10)
            .with_seed(42)
            .with_max_iters(5)
            .with_chunk_size_data(200)  // Small data chunk
            .with_chunk_size_centroids(3)  // Small centroid chunk
            .with_max_points_per_centroid(None);

        let mut kmeans = FastKMeansCuda::with_config(config).unwrap();
        let result = kmeans.train(&data.view());
        assert!(result.is_ok(), "Chunked CUDA training should succeed: {:?}", result.err());

        let labels = kmeans.predict(&data.view()).unwrap();
        assert_eq!(labels.len(), 1000);

        for &label in labels.iter() {
            assert!((0..10).contains(&label));
        }
    }
}

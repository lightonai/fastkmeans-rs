use crate::config::KMeansConfig;
use crate::distance::{
    compute_centroid_shift, compute_squared_norms, find_nearest_centroids_chunked,
};
use crate::error::KMeansError;
use ndarray::{Array1, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Result of the k-means algorithm
#[allow(dead_code)]
pub struct KMeansResult {
    pub centroids: Array2<f32>,
    pub labels: Array1<i64>,
    pub n_iterations: usize,
}

/// Run the double-chunked k-means algorithm
///
/// This algorithm processes both data and centroids in chunks to minimize
/// memory usage while maintaining efficiency.
pub fn kmeans_double_chunked(
    data: &ArrayView2<f32>,
    config: &KMeansConfig,
) -> Result<KMeansResult, KMeansError> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let k = config.k;

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
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    // Subsample if needed
    let (data_subset, _subset_indices) = subsample_data(data, config, &mut rng)?;
    let n_samples_used = data_subset.nrows();

    if config.verbose {
        eprintln!(
            "Training k-means: {} samples ({}), {} features, {} clusters",
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

    // Pre-compute data norms
    let data_norms = compute_squared_norms(&data_subset.view());

    // Initialize centroids (random selection from data)
    let mut centroids = initialize_centroids(&data_subset.view(), k, &mut rng);

    // Main k-means loop
    let mut labels = Array1::zeros(n_samples_used);
    let mut n_iterations = 0;

    for iteration in 0..config.max_iters {
        let iter_start = Instant::now();
        n_iterations = iteration + 1;

        // Pre-compute centroid norms
        let centroid_norms = compute_squared_norms(&centroids.view());

        // Accumulators for new centroids
        let mut cluster_sums: Array2<f32> = Array2::zeros((k, n_features));
        let mut cluster_counts: Array1<f32> = Array1::zeros(k);

        // Process data in chunks
        let mut start_idx = 0;
        while start_idx < n_samples_used {
            let end_idx = (start_idx + config.chunk_size_data).min(n_samples_used);
            let data_chunk = data_subset.slice(ndarray::s![start_idx..end_idx, ..]);
            let data_chunk_norms = data_norms.slice(ndarray::s![start_idx..end_idx]);

            // Find nearest centroids for this chunk
            let chunk_labels = find_nearest_centroids_chunked(
                &data_chunk,
                &data_chunk_norms,
                &centroids.view(),
                &centroid_norms.view(),
                config.chunk_size_centroids,
            );

            // Update labels
            for (i, &label) in chunk_labels.iter().enumerate() {
                labels[start_idx + i] = label;
            }

            // Accumulate cluster sums and counts
            for (i, &label) in chunk_labels.iter().enumerate() {
                let cluster_idx = label as usize;
                cluster_counts[cluster_idx] += 1.0;
                for j in 0..n_features {
                    cluster_sums[[cluster_idx, j]] += data_chunk[[i, j]];
                }
            }

            start_idx = end_idx;
        }

        // Compute new centroids
        let prev_centroids = centroids.clone();
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
                for j in 0..n_features {
                    centroids[[cluster_idx, j]] = data_subset[[data_idx, j]];
                }
            }

            if config.verbose {
                eprintln!("  Reinitialized {} empty clusters", empty_clusters.len());
            }
        }

        // Check convergence
        let shift = compute_centroid_shift(&prev_centroids.view(), &centroids.view());

        if config.verbose {
            let iter_time = iter_start.elapsed().as_secs_f64();
            eprintln!(
                "  Iteration {}/{}: shift = {:.6}, time = {:.4}s",
                iteration + 1,
                config.max_iters,
                shift,
                iter_time
            );
        }

        if config.tol >= 0.0 && shift < config.tol {
            if config.verbose {
                eprintln!(
                    "  Converged after {} iterations (shift {:.6} < tol {:.6})",
                    iteration + 1,
                    shift,
                    config.tol
                );
            }
            break;
        }
    }

    Ok(KMeansResult {
        centroids,
        labels,
        n_iterations,
    })
}

/// Subsample data if it exceeds the maximum size based on max_points_per_centroid
fn subsample_data(
    data: &ArrayView2<f32>,
    config: &KMeansConfig,
    rng: &mut ChaCha8Rng,
) -> Result<(Array2<f32>, Option<Vec<usize>>), KMeansError> {
    let n_samples = data.nrows();

    if let Some(max_ppc) = config.max_points_per_centroid {
        let max_samples = config.k * max_ppc;
        if n_samples > max_samples {
            if config.verbose {
                eprintln!(
                    "Subsampling data from {} to {} samples",
                    n_samples, max_samples
                );
            }

            // Random permutation and select first max_samples
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            indices.truncate(max_samples);
            indices.sort_unstable(); // Sort for cache-friendly access

            let n_features = data.ncols();
            let mut subset = Array2::zeros((max_samples, n_features));
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                for j in 0..n_features {
                    subset[[new_idx, j]] = data[[old_idx, j]];
                }
            }

            return Ok((subset, Some(indices)));
        }
    }

    // No subsampling needed - copy data
    Ok((data.to_owned(), None))
}

/// Initialize centroids by randomly selecting k data points
fn initialize_centroids(data: &ArrayView2<f32>, k: usize, rng: &mut ChaCha8Rng) -> Array2<f32> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    let indices: Vec<usize> = (0..n_samples).collect();
    let selected: Vec<usize> = indices.choose_multiple(rng, k).cloned().collect();

    let mut centroids = Array2::zeros((k, n_features));
    for (centroid_idx, &data_idx) in selected.iter().enumerate() {
        for j in 0..n_features {
            centroids[[centroid_idx, j]] = data[[data_idx, j]];
        }
    }

    centroids
}

/// Predict cluster assignments for new data using trained centroids
pub fn predict_labels(
    data: &ArrayView2<f32>,
    centroids: &ArrayView2<f32>,
    chunk_size_data: usize,
    chunk_size_centroids: usize,
) -> Array1<i64> {
    let n_samples = data.nrows();

    // Pre-compute norms
    let data_norms = compute_squared_norms(data);
    let centroid_norms = compute_squared_norms(centroids);

    let mut labels = Array1::zeros(n_samples);

    // Process in chunks
    let mut start_idx = 0;
    while start_idx < n_samples {
        let end_idx = (start_idx + chunk_size_data).min(n_samples);
        let data_chunk = data.slice(ndarray::s![start_idx..end_idx, ..]);
        let data_chunk_norms = data_norms.slice(ndarray::s![start_idx..end_idx]);

        let chunk_labels = find_nearest_centroids_chunked(
            &data_chunk,
            &data_chunk_norms,
            centroids,
            &centroid_norms.view(),
            chunk_size_centroids,
        );

        for (i, &label) in chunk_labels.iter().enumerate() {
            labels[start_idx + i] = label;
        }

        start_idx = end_idx;
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_initialize_centroids() {
        let data = Array2::random((100, 8), Uniform::new(-1.0f32, 1.0));
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let centroids = initialize_centroids(&data.view(), 5, &mut rng);

        assert_eq!(centroids.nrows(), 5);
        assert_eq!(centroids.ncols(), 8);
    }

    #[test]
    fn test_kmeans_basic() {
        let data = Array2::random((500, 16), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig {
            k: 5,
            max_iters: 10,
            tol: 1e-8,
            seed: 42,
            max_points_per_centroid: None,
            chunk_size_data: 51_200,
            chunk_size_centroids: 10_240,
            verbose: false,
        };

        let result = kmeans_double_chunked(&data.view(), &config).unwrap();

        assert_eq!(result.centroids.nrows(), 5);
        assert_eq!(result.centroids.ncols(), 16);
        assert_eq!(result.labels.len(), 500);

        // All labels should be valid
        for &label in result.labels.iter() {
            assert!((0..5).contains(&label));
        }
    }

    #[test]
    fn test_kmeans_with_subsampling() {
        let data = Array2::random((10000, 8), Uniform::new(-1.0f32, 1.0));

        let config = KMeansConfig {
            k: 10,
            max_iters: 5,
            tol: 1e-8,
            seed: 42,
            max_points_per_centroid: Some(256), // Will subsample to 2560
            chunk_size_data: 51_200,
            chunk_size_centroids: 10_240,
            verbose: false,
        };

        let result = kmeans_double_chunked(&data.view(), &config).unwrap();

        assert_eq!(result.centroids.nrows(), 10);
        // Labels are for the subsampled data
        assert_eq!(result.labels.len(), 2560);
    }
}

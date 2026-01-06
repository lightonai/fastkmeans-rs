use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Compute squared L2 norms for each row of a 2D array
/// Returns a 1D array where each element is the squared norm of the corresponding row
#[inline]
pub fn compute_squared_norms(data: &ArrayView2<f32>) -> Array1<f32> {
    let n_samples = data.nrows();
    let mut norms = Array1::zeros(n_samples);

    // Parallel computation of row norms
    norms
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, norm)| {
            let row = data.row(i);
            *norm = row.dot(&row);
        });

    norms
}

/// Compute squared L2 norms for each row of a 2D array (non-parallel version for small arrays)
#[inline]
#[allow(dead_code)]
pub fn compute_squared_norms_serial(data: &ArrayView2<f32>) -> Array1<f32> {
    let n_samples = data.nrows();
    let mut norms = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let row = data.row(i);
        norms[i] = row.dot(&row);
    }

    norms
}

/// Find the nearest centroid for each data point in a chunk using double-chunking
///
/// Uses the identity: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
///
/// # Arguments
/// * `data_chunk` - Chunk of data points (n_data, n_features)
/// * `data_norms` - Squared norms of data points (n_data,)
/// * `centroids` - All centroids (k, n_features)
/// * `centroid_norms` - Squared norms of centroids (k,)
/// * `chunk_size_centroids` - Size of centroid chunks
///
/// # Returns
/// * `labels` - Cluster assignments for each data point (n_data,)
pub fn find_nearest_centroids_chunked(
    data_chunk: &ArrayView2<f32>,
    data_norms: &ArrayView1<f32>,
    centroids: &ArrayView2<f32>,
    centroid_norms: &ArrayView1<f32>,
    chunk_size_centroids: usize,
) -> Array1<i64> {
    let n_data = data_chunk.nrows();
    let k = centroids.nrows();

    let mut best_labels = Array1::zeros(n_data);
    let mut best_dists = Array1::from_elem(n_data, f32::INFINITY);

    // Process centroids in chunks
    let mut c_start = 0;
    while c_start < k {
        let c_end = (c_start + chunk_size_centroids).min(k);
        let centroid_chunk = centroids.slice(ndarray::s![c_start..c_end, ..]);
        let centroid_chunk_norms = centroid_norms.slice(ndarray::s![c_start..c_end]);

        // Compute distances: ||x||^2 + ||c||^2 - 2*x.c
        // dist_chunk has shape (n_data, chunk_centroids)
        let n_centroids_chunk = c_end - c_start;

        // Compute x.c using matrix multiplication
        // data_chunk: (n_data, n_features), centroid_chunk.t(): (n_features, n_centroids_chunk)
        // Result: (n_data, n_centroids_chunk)
        let dot_products = data_chunk.dot(&centroid_chunk.t());

        // Parallel update of best labels and distances
        best_labels
            .as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .zip(best_dists.as_slice_mut().unwrap().par_iter_mut())
            .enumerate()
            .for_each(|(i, (label, best_dist))| {
                let x_norm = data_norms[i];

                for j in 0..n_centroids_chunk {
                    let c_norm = centroid_chunk_norms[j];
                    let dot = dot_products[[i, j]];

                    // Squared distance: ||x||^2 + ||c||^2 - 2*x.c
                    let dist = x_norm + c_norm - 2.0 * dot;

                    if dist < *best_dist {
                        *best_dist = dist;
                        *label = (c_start + j) as i64;
                    }
                }
            });

        c_start = c_end;
    }

    best_labels
}

/// Find the nearest centroid for each data point (serial version for small arrays)
#[allow(dead_code)]
pub fn find_nearest_centroids_serial(
    data_chunk: &ArrayView2<f32>,
    data_norms: &ArrayView1<f32>,
    centroids: &ArrayView2<f32>,
    centroid_norms: &ArrayView1<f32>,
) -> Array1<i64> {
    let n_data = data_chunk.nrows();
    let k = centroids.nrows();

    let mut best_labels = Array1::zeros(n_data);
    let mut best_dists = Array1::from_elem(n_data, f32::INFINITY);

    // Compute x.c using matrix multiplication
    let dot_products = data_chunk.dot(&centroids.t());

    for i in 0..n_data {
        let x_norm = data_norms[i];

        for j in 0..k {
            let c_norm = centroid_norms[j];
            let dot = dot_products[[i, j]];

            // Squared distance: ||x||^2 + ||c||^2 - 2*x.c
            let dist = x_norm + c_norm - 2.0 * dot;

            if dist < best_dists[i] {
                best_dists[i] = dist;
                best_labels[i] = j as i64;
            }
        }
    }

    best_labels
}

/// Compute centroid shift (sum of L2 norms of centroid movements)
pub fn compute_centroid_shift(
    old_centroids: &ArrayView2<f32>,
    new_centroids: &ArrayView2<f32>,
) -> f64 {
    let k = old_centroids.nrows();

    let shifts: f64 = (0..k)
        .into_par_iter()
        .map(|i| {
            let old_c = old_centroids.row(i);
            let new_c = new_centroids.row(i);

            let mut diff_sq = 0.0f64;
            for j in 0..old_c.len() {
                let d = (new_c[j] - old_c[j]) as f64;
                diff_sq += d * d;
            }
            diff_sq.sqrt()
        })
        .sum();

    shifts
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_compute_squared_norms() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let norms = compute_squared_norms(&data.view());

        assert_relative_eq!(norms[0], 1.0 + 4.0 + 9.0, epsilon = 1e-6);
        assert_relative_eq!(norms[1], 16.0 + 25.0 + 36.0, epsilon = 1e-6);
    }

    #[test]
    fn test_find_nearest_centroids() {
        // Simple 2D case
        let data = array![[0.0f32, 0.0], [10.0, 10.0], [5.0, 5.0]];
        let centroids = array![[0.0f32, 0.0], [10.0, 10.0]];

        let data_norms = compute_squared_norms(&data.view());
        let centroid_norms = compute_squared_norms(&centroids.view());

        let labels = find_nearest_centroids_serial(
            &data.view(),
            &data_norms.view(),
            &centroids.view(),
            &centroid_norms.view(),
        );

        assert_eq!(labels[0], 0); // (0,0) closest to centroid 0
        assert_eq!(labels[1], 1); // (10,10) closest to centroid 1
                                  // (5,5) is equidistant, but we take the first one found (0)
    }

    #[test]
    fn test_centroid_shift() {
        let old = array![[0.0f32, 0.0], [1.0, 1.0]];
        let new = array![[1.0f32, 0.0], [1.0, 1.0]];

        let shift = compute_centroid_shift(&old.view(), &new.view());
        assert_relative_eq!(shift, 1.0, epsilon = 1e-6);
    }
}

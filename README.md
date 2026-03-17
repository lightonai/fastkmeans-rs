<h1 align="center">fastkmeans-rs</h1>

<p align="center">
  <a href="https://crates.io/crates/fastkmeans-rs"><img src="https://img.shields.io/crates/v/fastkmeans-rs.svg" alt="Crates.io"></a>
  <a href="https://github.com/lightonai/fastkmeans-rs/blob/master/LICENSE"><img src="https://img.shields.io/crates/l/fastkmeans-rs.svg" alt="License"></a>
</p>

<p align="center">
  <strong>A fast Rust k-means with optional Metal GPU acceleration on macOS.</strong>
</p>

<br>

A Rust port of [FastKMeans](https://github.com/AnswerDotAI/fastkmeans) by Answer.AI with additional **Metal GPU acceleration** inspired by [flash-kmeans](https://github.com/svg-project/flash-kmeans). Uses double-chunking for memory-efficient large-scale clustering, multi-threaded CPU via `rayon`, and optional GPU offload via Metal Performance Shaders on Apple Silicon.

<br>

## Features

- **Metal GPU acceleration** — Enable `metal_gpu` on macOS for 1.5-1.7x speedup on medium-to-large workloads (Apple Silicon recommended)
- **Double-chunking algorithm** — Processes both data points and centroids in configurable chunks to avoid OOM issues on large datasets
- **Multi-threaded** — Leverages `rayon` for parallel computation across all CPU cores
- **Optional BLAS acceleration** — Enable `accelerate` (macOS) or `openblas` features for optimized matrix operations
- **CUDA GPU support** — Enable `cuda` for NVIDIA GPU acceleration with cuBLAS
- **ndarray compatible** — Seamlessly integrates with the Rust scientific computing ecosystem
- **Familiar API** — Provides both FAISS-style (`train`/`predict`) and scikit-learn-style (`fit`/`fit_predict`) interfaces
- **Memory efficient** — Constant memory usage regardless of dataset size through chunked processing
- **Reproducible** — Seeded random number generation for deterministic results

<br>

## Installation

Add `fastkmeans-rs` to your `Cargo.toml`:

```toml
[dependencies]
fastkmeans-rs = "0.1.3"
```

Or via cargo:

```bash
cargo add fastkmeans-rs
```

### With GPU / BLAS Acceleration (Recommended)

```toml
# macOS with Metal GPU (best performance on Apple Silicon)
fastkmeans-rs = { version = "0.1", features = ["metal_gpu", "accelerate"] }

# macOS CPU-only (uses Apple Accelerate BLAS)
fastkmeans-rs = { version = "0.1", features = ["accelerate"] }

# Linux/Windows with CUDA GPU
fastkmeans-rs = { version = "0.1", features = ["cuda"] }

# Linux/Windows CPU-only (requires OpenBLAS installed)
fastkmeans-rs = { version = "0.1", features = ["openblas"] }
```

When `metal_gpu` is enabled, `FastKMeans` automatically dispatches to the Metal GPU for workloads where it's beneficial (N&times;K &ge; 500K). Smaller workloads stay on CPU. No code changes needed.

<br>

## Benchmarks

### Accuracy Validation

fastkmeans-rs produces clustering results equivalent to the reference Python [fastkmeans](https://github.com/AnswerDotAI/fastkmeans) implementation. We validate this by comparing inertia (sum of squared distances to assigned centroids):

| Samples | Features | k | Python Inertia | Rust Inertia | Difference |
|---------|----------|---|----------------|--------------|------------|
| 1,000 | 64 | 10 | 59,630.96 | 59,816.55 | 0.31% |
| 500 | 32 | 5 | 14,693.69 | 14,588.72 | 0.71% |
| 2,000 | 128 | 20 | 242,046.80 | 242,268.88 | 0.09% |

The small differences (<1%) are expected since k-means converges to local minima based on random initialization.

Run the accuracy benchmark yourself:

```bash
uv run benches/compare_reference.py
```

### Metal GPU vs CPU (Apple M3 Pro)

100,000 points, 128 dimensions, 25 iterations:

| k | CPU (Accelerate BLAS) | Metal GPU | Speedup |
|---|---|---|---|
| 64 | 0.147s | 0.148s | 1.0x |
| 256 | 0.365s | 0.231s | **1.58x** |
| 512 | 0.622s | 0.397s | **1.57x** |
| 1,024 | 1.280s | 0.742s | **1.73x** |

Metal GPU uses MPS (Metal Performance Shaders) for optimized GEMM, single command buffer batching, and fused distance+argmin kernels. Memory usage is comparable to CPU thanks to Apple Silicon unified memory.

### CPU Performance vs Python

Benchmark comparing fastkmeans-rs (with BLAS) against Python fastkmeans (using PyTorch) on an Apple M3 Max:

| Config | Samples | Dims | k | Python | Rust | Speedup |
|--------|---------|------|---|--------|------|---------|
| Small | 1,000 | 64 | 10 | 0.006s | 0.003s | **2.16x** |
| Medium | 5,000 | 64 | 50 | 0.016s | 0.015s | **1.06x** |
| Large | 10,000 | 128 | 100 | 0.044s | 0.041s | **1.07x** |
| XL | 25,000 | 128 | 100 | 0.098s | 0.083s | **1.18x** |
| XXL | 50,000 | 128 | 256 | 0.315s | 0.293s | **1.08x** |

**Average speedup: 1.31x** (with BLAS enabled)

Run the performance benchmark yourself:

```bash
uv run benches/benchmark_comparison.py
```

> **Note:** Without BLAS, Rust performance is slower than Python on large datasets. Enable `accelerate` or `openblas` features for optimal performance.

<br>

## Quick Start

```rust
use fastkmeans_rs::FastKMeans;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    // Generate 10,000 random points in 128 dimensions
    let data = Array2::random((10_000, 128), Uniform::new(-1.0f32, 1.0));

    // Create k-means with 100 clusters
    let mut kmeans = FastKMeans::new(128, 100);

    // Train the model
    kmeans.train(&data.view()).unwrap();

    // Get cluster assignments
    let labels = kmeans.predict(&data.view()).unwrap();

    println!("Assigned {} points to clusters", labels.len());
}
```

<br>

## Usage

### Metal GPU (macOS)

With the `metal_gpu` feature enabled, `FastKMeans` automatically uses Metal GPU for large workloads — no code changes needed:

```rust
use fastkmeans_rs::FastKMeans;

let mut kmeans = FastKMeans::new(128, 256);
kmeans.train(&data.view()).unwrap();  // Uses Metal GPU automatically
let labels = kmeans.predict(&data.view()).unwrap();
```

You can also use the Metal backend directly for full control:

```rust
use fastkmeans_rs::metal_gpu::FastKMeansMetal;
use fastkmeans_rs::KMeansConfig;

let config = KMeansConfig::new(256)
    .with_max_iters(50)
    .with_verbose(true);

let mut kmeans = FastKMeansMetal::with_config(config).unwrap();
kmeans.train(&data.view()).unwrap();
let labels = kmeans.predict(&data.view()).unwrap();
```

### Basic Usage

```rust
use fastkmeans_rs::FastKMeans;
use ndarray::Array2;

// Your data as an ndarray (n_samples x n_features)
let data: Array2<f32> = /* your data */;

// Create a FastKMeans instance
// Arguments: dimensions (d), number of clusters (k)
let mut kmeans = FastKMeans::new(128, 50);

// Train the model
kmeans.train(&data.view())?;

// Predict cluster assignments for new data
let labels = kmeans.predict(&data.view())?;

// Access the learned centroids
if let Some(centroids) = kmeans.centroids() {
    println!("Centroid shape: {:?}", centroids.shape());
}
```

### Fit-Predict Pattern

For convenience, you can fit and predict in a single call:

```rust
use fastkmeans_rs::FastKMeans;

let mut kmeans = FastKMeans::new(128, 50);

// Fit and predict in one step
let labels = kmeans.fit_predict(&data.view())?;
```

### Custom Configuration

Fine-tune the algorithm with `KMeansConfig`:

```rust
use fastkmeans_rs::{FastKMeans, KMeansConfig};

let config = KMeansConfig {
    k: 100,                          // Number of clusters
    max_iters: 50,                   // Maximum iterations (default: 25)
    tol: 1e-6,                       // Convergence tolerance (default: 1e-8)
    seed: 42,                        // Random seed for reproducibility
    max_points_per_centroid: Some(256), // Subsampling threshold (None to disable)
    chunk_size_data: 51_200,         // Data chunk size for memory control
    chunk_size_centroids: 10_240,    // Centroid chunk size
    verbose: true,                   // Print progress information
};

let mut kmeans = FastKMeans::with_config(config);
kmeans.fit(&data.view())?;
```

### Configuration Parameters

| Parameter                 | Default   | Description                                   |
| ------------------------- | --------- | --------------------------------------------- |
| `k`                       | —         | Number of clusters (required)                 |
| `max_iters`               | 25        | Maximum number of iterations                  |
| `tol`                     | 1e-8      | Convergence tolerance based on centroid shift |
| `seed`                    | 0         | Random seed for reproducible results          |
| `max_points_per_centroid` | Some(256) | If set, subsamples data when `n > k * value`  |
| `chunk_size_data`         | 51,200    | Number of data points processed per chunk     |
| `chunk_size_centroids`    | 10,240    | Number of centroids processed per chunk       |
| `verbose`                 | false     | Print iteration progress                      |

<br>

## Algorithm

This implementation uses the **double-chunking k-means** algorithm from the original FastKMeans, with additional GPU acceleration inspired by [flash-kmeans](https://github.com/svg-project/flash-kmeans):

1. **Efficient distance computation** using the identity:

   ```
   ||x - c||² = ||x||² + ||c||² - 2·x·c
   ```

   This avoids materializing the full distance matrix by pre-computing norms and using matrix multiplication (BLAS GEMM on CPU, MPS GEMM on Metal GPU).

2. **Double-chunking** to control memory:

   - Outer loop: process data in chunks of `chunk_size_data`
   - Inner loop: process centroids in chunks of `chunk_size_centroids`
   - Memory usage is O(chunk_size_data × chunk_size_centroids) regardless of total dataset size

3. **Parallel processing** via rayon for:
   - Norm computation
   - Distance calculations
   - Cluster assignment updates
   - Centroid recomputation

4. **Metal GPU path** (when `metal_gpu` feature is enabled):
   - MPS (Metal Performance Shaders) for hardware-optimized GEMM
   - All GPU operations batched into a single command buffer per iteration
   - Fused distance + argmin kernel (no separate distance matrix)
   - Pre-allocated GPU buffers (zero allocation in the hot loop)
   - Unified memory on Apple Silicon (no CPU-GPU data transfers)

<br>

## Performance Tips

- **Enable Metal GPU on macOS** — `features = ["metal_gpu", "accelerate"]` gives the best performance on Apple Silicon
- **Adjust chunk sizes** based on your available memory. Larger chunks = faster but more memory
- **Use subsampling** (`max_points_per_centroid`) for very large datasets during initial exploration
- **Set `verbose: true`** to monitor convergence and iteration times
- **Compile with `--release`** for optimal performance (10-100x faster than debug builds)

<br>

## Example: Large-Scale Clustering

```rust
use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate 1 million points in 256 dimensions
    let n_samples = 1_000_000;
    let n_features = 256;
    let n_clusters = 1000;

    println!("Generating {} points...", n_samples);
    let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));

    let config = KMeansConfig {
        k: n_clusters,
        max_iters: 25,
        tol: 1e-6,
        seed: 42,
        max_points_per_centroid: Some(256),  // Subsample to 256k points
        chunk_size_data: 50_000,
        chunk_size_centroids: 500,
        verbose: true,
    };

    let mut kmeans = FastKMeans::with_config(config);

    println!("Training k-means with {} clusters...", n_clusters);
    kmeans.train(&data.view())?;

    println!("Predicting labels...");
    let labels = kmeans.predict(&data.view())?;

    println!("Done! Assigned {} points to {} clusters", labels.len(), n_clusters);

    Ok(())
}
```

<br>

## Contributing

### Setup

1. Clone the repository:

```bash
git clone https://github.com/lightonai/fastkmeans-rs.git
cd fastkmeans-rs
```

2. Install the git hooks:

```bash
make install-hooks
```

### Development Commands

| Command                  | Description                                                                       |
| ------------------------ | --------------------------------------------------------------------------------- |
| `make build`             | Build the project in debug mode                                                   |
| `make release`           | Build in release mode                                                             |
| `make test`              | Run all tests                                                                     |
| `make lint`              | Run clippy and format checks                                                      |
| `make fmt`               | Format code                                                                       |
| `make doc`               | Build documentation                                                               |
| `make bench`             | Run benchmarks                                                                    |
| `make example`              | Run the basic example                                                             |
| `make ci`                   | Run all CI checks locally                                                         |
| `make build-blas`           | Build with BLAS acceleration (auto-detects platform)                              |
| `make test-blas`            | Run tests with BLAS acceleration                                                  |
| `make compare-reference`    | Compare output with Python fastkmeans (requires [uv](https://docs.astral.sh/uv/)) |
| `make benchmark-comparison` | Run performance comparison vs Python fastkmeans                                   |

### Running CI Locally

Before submitting a PR, run all checks locally:

```bash
make ci
```

This runs formatting checks, clippy, tests, documentation build, and benchmark compilation.

<br>

## Acknowledgements

This crate is a Rust port of [FastKMeans](https://github.com/AnswerDotAI/fastkmeans) by Answer.AI, with Metal GPU acceleration inspired by [flash-kmeans](https://github.com/svg-project/flash-kmeans) and [flash-kmeans-mlx](https://github.com/hanxiao/flash-kmeans-mlx). Credit for the algorithm design goes to the original authors.

<br>

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.

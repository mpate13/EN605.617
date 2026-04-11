import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# --- 1. CIFAR-10 DATA LOADER ---
def load_cifar10_batch(file_path, num_images=5000):
    """Parses the CIFAR-10 binary format."""
    with open(file_path, 'rb') as f:
        # Each image is 3073 bytes: 1 byte label + 3072 bytes pixels
        raw_data = np.frombuffer(f.read(num_images * 3073), dtype=np.uint8)
    
    # Reshape and remove the label byte (first column)
    reshaped = raw_data.reshape(num_images, 3073)
    pixels = reshaped[:, 1:].astype(np.float32) / 255.0  # Normalize to [0,1]
    return pixels

# --- 2. CUDA KERNEL DEFINITION ---
# This matches your assignment.cu logic but optimized for Python orchestration
kmeans_kernel = cp.RawKernel(r'''
extern "C" __global__
void assign_clusters(const float* data, const float* centroids, int* assignments, 
                     int n, int k, int dims, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* current_point = data + ((idx + offset) * dims);
    float min_dist = 1e38; 
    int best_cluster = 0;

    for (int i = 0; i < k; i++) {
        float dist = 0.0;
        for (int d = 0; d < dims; d++) {
            float diff = current_point[d] - centroids[i * dims + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = i;
        }
    }
    assignments[idx + offset] = best_cluster;
}
''', 'assign_clusters')

# --- 3. K-MEANS SOLVER ---
def run_kmeans(data_host, k=10, max_iter=20, batch_size=None):
    n, dims = data_host.shape
    data_gpu = cp.asarray(data_host)
    # Start with random centroids from the dataset
    centroids_gpu = data_gpu[cp.random.choice(n, k, replace=False)]
    assignments_gpu = cp.zeros(n, dtype=cp.int32)

    # Determine if we use full dataset or mini-batch
    actual_batch = batch_size if batch_size and batch_size < n else n
    t_pb = 256
    b_pg = (actual_batch + t_pb - 1) // t_pb

    print(f"Clustering {n} images in {'Mini-Batch' if actual_batch < n else 'Standard'} mode...")

    for i in range(max_iter):
        offset = np.random.randint(0, n - actual_batch + 1) if actual_batch < n else 0
        
        # 1. Assignment Phase (CUDA)
        kmeans_kernel((b_pg,), (t_pb,), (data_gpu, centroids_gpu, assignments_gpu, actual_batch, k, dims, offset))
        
        # 2. Update Phase (Centroid Re-calculation)
        new_centroids = cp.zeros_like(centroids_gpu)
        for j in range(k):
            mask = (assignments_gpu == j)
            count = cp.sum(mask)
            if count > 0:
                new_centroids[j] = cp.sum(data_gpu[mask], axis=0) / count
        centroids_gpu = new_centroids

    return assignments_gpu.get(), data_host

# --- 4. VISUALIZATION (PCA) ---
def plot_results(data, assignments):
    print("Reducing 3072 dimensions to 2D for visualization...")
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=assignments, cmap='tab10', s=2, alpha=0.6)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("CIFAR-10 Clustering Visualized via PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    cifar_file = "data_batch_1.bin" # Ensure this file exists
    
    if os.path.exists(cifar_file):
        images = load_cifar10_batch(cifar_file, num_images=10000)
        # Run with optional batch size (e.g., 2048) or remove for full K-means
        results, original_data = run_kmeans(images, k=10, batch_size=2048)
        plot_results(original_data, results)
    else:
        print(f"Error: {cifar_file} not found. Please download the CIFAR-10 binary version.")
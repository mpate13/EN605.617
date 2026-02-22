#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

/**
 * K-Means Clustering: Accelerated Assignment Phase
 * ------------------------------------------------------------------
 * This program implements the most computationally intensive portion of the 
 * K-Means algorithm: the "Assignment Phase" (also known as Expectation).
 * * HOW IT RELATES TO K-MEANS:
 * K-Means is an iterative 2-step process:
 * 1. Assignment (THIS CODE): Every data point finds its nearest centroid.
 * 2. Update (Next Step): Centroids move to the average of their assigned 
 * points. This GPU kernel handles Step 1. In a full clustering application, 
 * this code would run inside a loop, repeating until the centroids "converge" 
 * (stop moving).
 * By accelerating this phase, we optimize the part of the algorithm that 
 * scales linearly with the number of data points, 
 * which is where CPU bottlenecks occur.
 *
 * GPU ARCHITECTURE OPTIMIZATIONS:
 * - Constant Memory: Centroids are cached for fast "broadcast" reads.
 * - Shared Memory: Points are staged in a block-level "scratchpad" to hide
 *  VRAM latency.
 * - Register Math: Individual thread calculations happen at peak 
 * hardware speed.
 * - Validation: Results are verified against a serial CPU baseline 
 * for accuracy.
 */


#define MAX_CLUSTERS 16

/**
 * CONSTANT MEMORY: 
 * Stores cluster centroids. This is cached and optimized for 
 * "broadcast" reads where every thread reads the same value.
 */
__constant__ float c_centroids_x[MAX_CLUSTERS];
__constant__ float c_centroids_y[MAX_CLUSTERS];

/**
 * The memory path, as discussed in lecture
 * Path: Global -> Shared -> Registers -> Global (with constant lookups)
 */
__global__ void kmeans_gpu_kernel(float *d_x, float *d_y, int *d_cluster, 
                                   int n_points, int k) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Check threads don't access memory outside of n_points
    if (tid >= n_points) return;

    /**
     * SHARED MEMORY:
     * Shared by threads within the same block.
     * Partitioned into two arrays for X and Y coordinates.
     */
    extern __shared__ float s_data[]; 
    float *s_x = &s_data[0];
    float *s_y = &s_data[blockDim.x];

    // Load from Global Memory to Shared Memory
    s_x[local_id] = d_x[tid];
    s_y[local_id] = d_y[tid];

    // Synchronize to ensure all threads have finished loading 
    // before math begins
    __syncthreads(); 

    /**
     * REGISTERS:
     * Thread-private variables used for the fast math
     */
    float px = s_x[local_id];
    float py = s_y[local_id];
    float min_dist = FLT_MAX;
    int best_cluster = -1;

    // Calculate distances using Registers and 
    // Constant Memory Centroids (lookups)
    for (int i = 0; i < k; i++) {
        float dx = px - c_centroids_x[i];
        float dy = py - c_centroids_y[i];
        float dist = (dx * dx) + (dy * dy); // squared euclidean distance

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = i;
        }
    }

    // Write final assignment back to Global Memory
    d_cluster[tid] = best_cluster;
}

// Host CPU implementation for validation and speedup comparison
void kmeans_cpu(float *x, float *y, int *cluster, float *cx, 
    float *cy, int n, int k) {
    for (int i = 0; i < n; i++) {
        float min_dist = FLT_MAX;
        int best_cluster = -1;
        for (int j = 0; j < k; j++) {
            float dx = x[i] - cx[j];
            float dy = y[i] - cy[j];
            float dist = (dx * dx) + (dy * dy);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        cluster[i] = best_cluster;
    }
}

void run_benchmark(int blocks, int threads) {
    // Querying actual hardware for some of safety checks... 
    // (ensure there are not more threads than the limit, etc)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Hardware limit checks (safety)
    if (threads > prop.maxThreadsPerBlock || threads <= 0) {
        printf("Error: Threads per block (%d) exceeds GPU limit (%d).\n", 
            threads, prop.maxThreadsPerBlock);
        return;
    }

    // (Number of Threads) * (4 bytes per float) * (2 arrays: X and Y).
    // This size is passed to the kernel
    size_t shared_mem_size = threads * sizeof(float) * 2; 
    if (shared_mem_size > prop.sharedMemPerBlock) {
        printf("Error: Shared memory (%.1f KB) exceeds limit (%.1f KB).\n", 
               shared_mem_size/1024.0, prop.sharedMemPerBlock/1024.0);
        return;
    }

    int n_points = blocks * threads;
    int k = 8;
    size_t f_size = n_points * sizeof(float);
    size_t i_size = n_points * sizeof(int);

    // Host Allocation
    float *h_x = (float*)malloc(f_size);
    float *h_y = (float*)malloc(f_size);
    int *h_cluster_gpu = (int*)malloc(i_size);
    int *h_cluster_cpu = (int*)malloc(i_size);
    float h_cx[MAX_CLUSTERS], h_cy[MAX_CLUSTERS];

    srand(time(NULL));
    for(int i=0; i<n_points; i++) { 
        h_x[i] = (float)(rand()%100); h_y[i] = (float)(rand()%100); 
    }
    for(int i=0; i<k; i++) { 
        h_cx[i] = (float)(rand()%100); h_cy[i] = (float)(rand()%100); 
    }

    // CPU Baseline Timing
    clock_t cpu_start = clock();
    kmeans_cpu(h_x, h_y, h_cluster_cpu, h_cx, h_cy, n_points, k);
    clock_t cpu_stop = clock();
    double cpu_ms = ((double)(cpu_stop - cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // GPU Allocation & Error Checking
    float *d_x = NULL, *d_y = NULL;
    int *d_cluster = NULL;
    if (cudaMalloc(&d_x, f_size) != cudaSuccess || 
        cudaMalloc(&d_y, f_size) != cudaSuccess || 
        cudaMalloc(&d_cluster, i_size) != cudaSuccess) {
        printf("Error: GPU Memory Allocation failed (Global Memory).\n");
        if(d_x) cudaFree(d_x); if(d_y) cudaFree(d_y);
        free(h_x); free(h_y); free(h_cluster_gpu); free(h_cluster_cpu);
        return;
    }

    // Transfer global to constant memory
    cudaMemcpy(d_x, h_x, f_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, f_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_centroids_x, h_cx, k * sizeof(float));
    cudaMemcpyToSymbol(c_centroids_y, h_cy, k * sizeof(float));

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel execution
    kmeans_gpu_kernel<<<blocks, threads, shared_mem_size>>>(d_x, d_y, 
        d_cluster, n_points, k);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    } else {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms;
        cudaEventElapsedTime(&gpu_ms, start, stop);
        cudaMemcpy(h_cluster_gpu, d_cluster, i_size, cudaMemcpyDeviceToHost);

        // GPU vs CPU result comparison
        // check to see if the rsults are actually the same
        int match_count = 0;
        for (int i = 0; i < n_points; i++) {
            if (h_cluster_gpu[i] == h_cluster_cpu[i]) {
                match_count++;
            }
        }

        printf("\n--- Results: %d Blocks x %d Threads ---\n", blocks, threads);
        printf("Hardware:    %s\n", prop.name);
        printf("Validation:  %d/%d points matched CPU results.\n", 
            match_count, n_points);
        printf("Host Time:   %10.4f ms\n", cpu_ms);
        printf("Device Time: %10.4f ms\n", gpu_ms);
        printf("Speedup:     %10.2fx\n", cpu_ms / gpu_ms);
    }

    // Clean up
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_cluster);
    free(h_x); free(h_y); free(h_cluster_gpu); free(h_cluster_cpu);
}

int main(int argc, char** argv) {
    if (argc < 3 || atoi(argv[1]) <= 0 || atoi(argv[2]) <= 0) {
        printf("Usage: ./assignment <blocks> <threads>\n");
        return 1;
    }
    run_benchmark(atoi(argv[1]), atoi(argv[2]));
    return 0;
}
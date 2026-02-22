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
 * Logic for calculating the closest centroid for a single point.
 * Uses constant memory for centroids and registers for math.
 */
__device__ int find_nearest_cluster(float px, float py, int k)
{
    float min_dist = FLT_MAX;
    int best_cluster = -1;

    for (int i = 0; i < k; i++)
    {
        float dx = px - c_centroids_x[i];
        float dy = py - c_centroids_y[i];
        float dist = (dx * dx) + (dy * dy);

        if (dist < min_dist)
        {
            min_dist = dist;
            best_cluster = i;
        }
    }
    return best_cluster;
}

/**
 * Handles memory staging and thread indexing.
 */
__global__ void kmeans_gpu_kernel(float *d_x, float *d_y, int *d_cluster, 
                                   int n_points, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    if (tid >= n_points)
    {
        return;
    }

    // Shared Memory Staging
    extern __shared__ float s_data[];
    float *s_x = &s_data[0];
    float *s_y = &s_data[blockDim.x];

    s_x[local_id] = d_x[tid];
    s_y[local_id] = d_y[tid];

    __syncthreads();

    // Call the device function for the actual calculation
    int assignment = find_nearest_cluster(s_x[local_id], s_y[local_id], k);

    d_cluster[tid] = assignment;
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

/**
 * Helper to ensure requested threads and shared memory fit the hardware.
 */
int validate_hardware(int threads, size_t shared_mem_size)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (threads > prop.maxThreadsPerBlock || threads <= 0)
    {
        printf("Error: Threads per block (%d) exceeds limit (%d).\n", 
            threads, prop.maxThreadsPerBlock);
        return 0;
    }

    if (shared_mem_size > prop.sharedMemPerBlock)
    {
        printf("Error: Shared memory (%.1f KB) exceeds limit (%.1f KB).\n", 
            shared_mem_size / 1024.0, prop.sharedMemPerBlock / 1024.0);
        return 0;
    }
    return 1;
}

/**
 * Handles GPU allocation, data transfer, kernel launch, and timing.
 */
void execute_gpu_workflow(float *h_x, float *h_y, int *h_gpu_res, 
    int n, int k, int blks, int thr, size_t smem)
{
    float *d_x, *d_y;
    int *d_c;
    cudaEvent_t start, stop;
    float gpu_ms;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(int));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kmeans_gpu_kernel<<<blks, thr, smem>>>(d_x, d_y, d_c, n, k);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_gpu_res, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Device Time: %10.4f ms\n", gpu_ms);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_c);
}

/**
 * Helper to initialize data and centroids.
 */
void init_data(float *x, float *y, float *cx, float *cy, int n, int k)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = (float)(rand() % 100);
        y[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < k; i++)
    {
        cx[i] = (float)(rand() % 100);
        cy[i] = (float)(rand() % 100);
    }
}

/**
 * Helper to compare results and free host memory.
 */
void verify_and_free(float *x, float *y, int *gpu, int *cpu, int n)
{
    int matches = 0;
    for (int i = 0; i < n; i++)
    {
        if (gpu[i] == cpu[i])
        {
            matches++;
        }
    }
    printf("Validation:  %d/%d matched.\n", matches, n);
    free(x);
    free(y);
    free(gpu);
    free(cpu);
}

/**
 * Main benchmark runner
 */
void run_benchmark(int blocks, int threads)
{
    int n = blocks * threads;
    int k = 8;
    size_t smem = threads * sizeof(float) * 2;
    float h_cx[MAX_CLUSTERS], h_cy[MAX_CLUSTERS];

    if (!validate_hardware(threads, smem))
    {
        return;
    }

    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    int *h_c_gpu = (int*)malloc(n * sizeof(int));
    int *h_c_cpu = (int*)malloc(n * sizeof(int));

    init_data(h_x, h_y, h_cx, h_cy, n, k);
    cudaMemcpyToSymbol(c_centroids_x, h_cx, k * sizeof(float));
    cudaMemcpyToSymbol(c_centroids_y, h_cy, k * sizeof(float));

    kmeans_cpu(h_x, h_y, h_c_cpu, h_cx, h_cy, n, k);
    execute_gpu_workflow(h_x, h_y, h_c_gpu, n, k, blocks, threads, smem);

    verify_and_free(h_x, h_y, h_c_gpu, h_c_cpu, n);
}

int main(int argc, char** argv) {
    if (argc < 3 || atoi(argv[1]) <= 0 || atoi(argv[2]) <= 0) {
        printf("Usage: ./assignment <blocks> <threads>\n");
        return 1;
    }
    run_benchmark(atoi(argv[1]), atoi(argv[2]));
    return 0;
}
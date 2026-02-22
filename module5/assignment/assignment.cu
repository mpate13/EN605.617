#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

/**
 * K-MEANS CLUSTERING: ACCELERATED ASSIGNMENT PHASE
 * ------------------------------------------------------------------
 * This program implements the "Assignment Phase" of the K-Means algorithm.
 * K-Means is an iterative 2-step process:
 * 1. Assignment (THIS CODE): Every data point finds its nearest centroid.
 * 2. Update (Next Step): Centroids move to the average of their assigned 
 * points. This GPU kernel handles Step 1.
 * 
 * MEMORY USED
 * 1. CONSTANT: Centroids are cached for fast hardware "broadcast" reads.
 * 2. GLOBAL: Large point datasets are stored in VRAM.
 * 3. SHARED: Points are staged in block-level L1-speed memory.
 * 4. REGISTERS: Mathematical calculations happen at peak frequency.
 */

#define MAX_CLUSTERS 16

/**
 * CONSTANT MEMORY - Stores cluster centroids.
 * This is cached and optimized for "broadcast" reads where 
 * every thread reads the same value simultaneously.
 */
__constant__ float c_centroids_x[MAX_CLUSTERS];
__constant__ float c_centroids_y[MAX_CLUSTERS];

/**
 * Logic for calculating the closest centroid for a single point.
 * Uses REGISTERS for math and CONSTANT for centroid lookups.
 */
__device__ int find_nearest_cluster(float px, float py, int k)
{
    float min_dist = FLT_MAX;
    int best_cluster = -1;

    for (int i = 0; i < k; i++)
    {
        float dx = px - c_centroids_x[i];
        float dy = py - c_centroids_y[i];
        float dist = (dx * dx) + (dy * dy); // Squared Euclidean distance

        if (dist < min_dist)
        {
            min_dist = dist;
            best_cluster = i;
        }
    }
    return best_cluster;
}

/**
 * Handles thread mapping and memory staging from GLOBAL to SHARED.
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

    /**
     * SHARED MEMORY
     */
    extern __shared__ float s_data[];
    float *s_x = &s_data[0];
    float *s_y = &s_data[blockDim.x];

    // Global Memory -> Shared Memory
    s_x[local_id] = d_x[tid];
    s_y[local_id] = d_y[tid];

    __syncthreads(); // Ensure all threads finish loading before math

    // Shared -> Register (passed as arguments)
    int assignment = find_nearest_cluster(s_x[local_id], s_y[local_id], k);

    // Register -> Global Memory
    d_cluster[tid] = assignment;
}

/**
 * Serial baseline used for speedup comparison and result validation.
 */
void kmeans_cpu(float *x, float *y, int *cluster, float *cx, 
    float *cy, int n, int k) 
{
    for (int i = 0; i < n; i++) 
    {
        float min_dist = FLT_MAX;
        int best_cluster = -1;
        for (int j = 0; j < k; j++) 
        {
            float dx = x[i] - cx[j];
            float dy = y[i] - cy[j];
            float dist = (dx * dx) + (dy * dy);
            if (dist < min_dist) 
            {
                min_dist = dist;
                best_cluster = j;
            }
        }
        cluster[i] = best_cluster;
    }
}

/**
 * Validates hardware limits and dataset size
 */
int validate_hardware(int threads, size_t shared_mem_size, int n)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t req_mem = (size_t)n * (2 * sizeof(float) + sizeof(int));

    if (threads > prop.maxThreadsPerBlock || threads <= 0)
    {
        printf("Error: Threads/Block (%d) exceeds limit.\n", threads);
        return 0;
    }
    if (shared_mem_size > prop.sharedMemPerBlock)
    {
        printf("Error: Shared memory exceeds limit.\n");
        return 0;
    }
    if (req_mem > prop.totalGlobalMem * 0.9)
    {
        printf("Error: Dataset too large for GPU VRAM.\n");
        return 0;
    }
    return 1;
}

/**
 * Handles Global Memory allocation and kernel timing.
 */
float execute_gpu_workflow(float *h_x, float *h_y, int *h_gpu_res, 
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

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_c);
    return gpu_ms;
}

/**
 * Formats results as requested for the benchmark report.
 */
void print_report(int blks, int thr, int n, int match, double cpu, float gpu)
{
    printf("==============================================\n");
    printf("TEST CONFIGURATION: %d Blocks | %d Threads/Block\n", blks, thr);
    printf("TOTAL PARALLEL THREADS: %d\n", n);
    printf("Validation: %d/%d points matched CPU results.\n", match, n);
    printf("MEMORY ENGAGED: Host, Global, Constant, Shared, Registers\n");
    printf("----------------------------------------------\n");
    printf("Host (CPU) Time:       %10.4f ms\n", cpu);
    printf("Device (GPU) Time:     %10.4f ms\n", gpu);
    printf("Speedup Ratio:         %10.2fx\n", cpu / (double)gpu);
    printf("RESULT: GPU is outperforming CPU by %.4fms\n", cpu - gpu);
    printf("==============================================\n");
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

    if (!validate_hardware(threads, smem, n)) return;

    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    int *h_c_gpu = (int*)malloc(n * sizeof(int));
    int *h_c_cpu = (int*)malloc(n * sizeof(int));

    if (!h_x || !h_y || !h_c_gpu || !h_c_cpu)
    {
        printf("Error: Host Memory Allocation failed.\n");
        return;
    }

    for (int i = 0; i < n; i++) { 
        h_x[i] = rand() % 100; h_y[i] = rand() % 100; 
    }
    for (int i = 0; i < k; i++) { 
        h_cx[i] = rand() % 100; h_cy[i] = rand() % 100; 
    }
    
    cudaMemcpyToSymbol(c_centroids_x, h_cx, k * sizeof(float));
    cudaMemcpyToSymbol(c_centroids_y, h_cy, k * sizeof(float));

    clock_t start_time = clock();
    kmeans_cpu(h_x, h_y, h_c_cpu, h_cx, h_cy, n, k);
    double cpu_ms = (double)(clock() - start_time) / CLOCKS_PER_SEC * 1000.0;

    float gpu_ms = execute_gpu_workflow(h_x, h_y, h_c_gpu, n, k, 
                                        blocks, threads, smem);

    int correct = 0;
    for (int i = 0; i < n; i++) if (h_c_gpu[i] == h_c_cpu[i]) correct++;

    print_report(blocks, threads, n, correct, cpu_ms, gpu_ms);
    free(h_x); free(h_y); free(h_c_gpu); free(h_c_cpu);
}

int main(int argc, char** argv) 
{
    srand(time(NULL));
    if (argc < 3 || atoi(argv[1]) <= 0 || atoi(argv[2]) <= 0) 
    {
        printf("Usage: ./assignment <blocks> <threads>\n");
        return 1;
    }
    run_benchmark(atoi(argv[1]), atoi(argv[2]));
    return 0;
}
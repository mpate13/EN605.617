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
 * 1. Assignment (THIS CODE): Every data point finds its nearest centroid via
 * Euclidean distance.
 * 2. Update (Next Step): Centroids are recalculated as the mean of assigned
 * points. This GPU kernel handles Step 1.
 * 
 * MEMORY USED:
 * 1. CONSTANT MEMORY: Centroids are cached here. This is optimized for 
 * "broadcast" reads where every thread in a warp accesses the same data.
 * 2. GLOBAL MEMORY: The bulk point dataset and final assignments are stored
 * in VRAM for high-capacity throughput.
 * 3. SHARED MEMORY: Points are staged from Global to this block-level 
 * L1-speed scratchpad to minimize high-latency Global Memory trips.
 * 4. REGISTERS: All mathematical calculations (distance, local IDs, loops)
 * are performed at peak frequency within the CUDA core.
 * 5. HOST MEMORY
 */

#define MAX_CLUSTERS 16

// CONSTANT MEMORY
__constant__ float c_centroids_x[MAX_CLUSTERS];
__constant__ float c_centroids_y[MAX_CLUSTERS];

/**
 * Logic for calculating the closest centroid for a single point.
 * Uses REGISTERS for local math and CONSTANT MEMORY for centroid lookups.
 */
__device__ int find_nearest_cluster(float px, float py, int k) {
    float min_dist = FLT_MAX;
    int best_cluster = -1;

    for (int i = 0; i < k; i++) {
        float dx = px - c_centroids_x[i];
        float dy = py - c_centroids_y[i];
        float dist = (dx * dx) + (dy * dy);

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = i;
        }
    }
    return best_cluster;
}

/**
 * Handles thread mapping and memory 
 * staging from GLOBAL MEMORY to SHARED MEMORY.
 */
__global__ void kmeans_gpu_kernel(float *d_x, float *d_y, int *d_cluster, 
                                   int n_points, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    if (tid >= n_points) return;

    // SHARED MEMORY
    extern __shared__ float s_data[];
    float *s_x = &s_data[0];
    float *s_y = &s_data[blockDim.x];

    // Global -> Shared
    s_x[local_id] = d_x[tid];
    s_y[local_id] = d_y[tid];

    __syncthreads();

    // Shared -> Register
    int assignment = find_nearest_cluster(s_x[local_id], s_y[local_id], k);

    // Register -> Global
    d_cluster[tid] = assignment;
}

/**
 * Serial baseline used for speedup comparison and result validation.
 */
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
 * Validates hardware limits and dataset size.
 */
int validate_hardware(int threads, size_t shared_mem_size, int n) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t req_mem = (size_t)n * (2 * sizeof(float) + sizeof(int));

    if (threads > prop.maxThreadsPerBlock || threads <= 0) {
        printf("Error: Threads/Block (%d) exceeds limit.\n", threads);
        return 0;
    }
    if (shared_mem_size > prop.sharedMemPerBlock) {
        printf("Error: Shared memory exceeds limit.\n");
        return 0;
    }
    if (req_mem > prop.totalGlobalMem * 0.9) {
        printf("Error: Dataset too large for GPU GLOBAL MEMORY.\n");
        return 0;
    }
    return 1;
}

/**
 * Checks for kernel launch and execution errors.
 */
void check_cuda_errors(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
}

/**
 * Manages GLOBAL MEMORY allocation and kernel timing.
 */
float execute_gpu_workflow(float *h_x, float *h_y, int *h_gpu_res, 
    int n, int k, int blks, int thr, size_t smem) {
    float *d_x, *d_y, gpu_ms;
    int *d_c;
    cudaEvent_t start, stop;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(int));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    kmeans_gpu_kernel<<<blks, thr, smem>>>(d_x, d_y, d_c, n, k);
    check_cuda_errors("Kernel Launch");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    cudaMemcpy(h_gpu_res, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_c);
    return gpu_ms;
}

/**
 * Initializes data points and centroids on the HOST MEMORY.
 */
void init_host_data(float *x, float *y, float *cx, float *cy, int n, int k) {
    for (int i = 0; i < n; i++) { 
        x[i] = (float)(rand() % 100); 
        y[i] = (float)(rand() % 100); 
    }
    for (int i = 0; i < k; i++) { 
        cx[i] = (float)(rand() % 100); 
        cy[i] = (float)(rand() % 100); 
    }
}

/**
 * Executes the CPU baseline and returns timing in ms.
 */
double run_cpu_baseline(float *x, float *y, int *res, float *cx, 
    float *cy, int n, int k) {
    clock_t start_time = clock();
    kmeans_cpu(x, y, res, cx, cy, n, k);
    return (double)(clock() - start_time) / CLOCKS_PER_SEC * 1000.0;
}

/**
 * Formats results as requested for the benchmark report.
 */
void print_report(int blks, int thr, int n, int match, double cpu, float gpu) {
    printf("==============================================\n");
    printf("TEST CONFIGURATION: %d Blocks | %d Threads/Block\n", blks, thr);
    printf("TOTAL PARALLEL THREADS: %d\n", n);
    printf("Validation: %d/%d points matched CPU results.\n", match, n);
    printf("----------------------------------------------\n");
    printf("Host (CPU) Time:       %10.4f ms\n", cpu);
    printf("Device (GPU) Time:     %10.4f ms\n", gpu);
    printf("Speedup Ratio:         %10.2fx\n", cpu / (double)gpu);
    printf("==============================================\n");
}

/**
 * Core execution logic: Initializing data, performing the 
 * K-Means runs (CPU vs GPU), and validating results.
 */
void execute_test_logic(int n, int k, int blocks, int threads, 
                        size_t smem, float *h_x, float *h_y, 
                        int *h_c_gpu, int *h_c_cpu) {
    float h_cx[MAX_CLUSTERS], h_cy[MAX_CLUSTERS];

    // 1. Initialize data and setup Constant Memory
    init_host_data(h_x, h_y, h_cx, h_cy, n, k);
    
    if (cudaMemcpyToSymbol(c_centroids_x, h_cx, k * sizeof(float)) != 
        cudaSuccess ||
        cudaMemcpyToSymbol(c_centroids_y, h_cy, k * sizeof(float)) != 
        cudaSuccess) {
        printf("Error: CONSTANT MEMORY Copy Failed.\n");
        return;
    }

    // 2. Perform Benchmarking
    double cpu_ms = run_cpu_baseline(h_x, h_y, h_c_cpu, h_cx, h_cy, n, k);
    float gpu_ms = execute_gpu_workflow(h_x, h_y, h_c_gpu, 
                                        n, k, blocks, threads, smem);

    // 3. Validate results and print report
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (h_c_gpu[i] == h_c_cpu[i]) correct++;
    }

    print_report(blocks, threads, n, correct, cpu_ms, gpu_ms);
}

/**
 * Validates hardware, manages Host Memory lifecycle, 
 * and calls the test logic.
 */
void run_benchmark(int blocks, int threads, int totalPoints) {
    int n = totalPoints;
    int k = 8;
    size_t smem = (size_t)threads * sizeof(float) * 2;

    // Hardware Limit Check
    if (!validate_hardware(threads, smem, n)) return;

    // HOST MEMORY Allocation
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    int *h_c_gpu = (int*)malloc(n * sizeof(int));
    int *h_c_cpu = (int*)malloc(n * sizeof(int));

    if (!h_x || !h_y || !h_c_gpu || !h_c_cpu) {
        printf("Error: Host Memory Allocation Failed.\n");
        return;
    }

    // Call the execution logic
    execute_test_logic(n, k, blocks, threads, smem, h_x, h_y, h_c_gpu, h_c_cpu);

    // Clean up HOST MEMORY
    free(h_x); 
    free(h_y); 
    free(h_c_gpu); 
    free(h_c_cpu);
}

int main(int argc, char** argv) {
    srand(time(NULL));
    if (argc < 3) {
        printf("Usage: ./assignment <total_threads> <block_size>\n");
        return 1;
    }

    int totalThreads = atoi(argv[1]);
    int blockSize = atoi(argv[2]);

    if (totalThreads <= 0 || blockSize <= 0) return 1;

    // Ceiling division to find number of blocks
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // Pass the actual totalThreads as a third argument
    run_benchmark(numBlocks, blockSize, totalThreads); 
    return 0;
}
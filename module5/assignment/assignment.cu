#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <time.h>

#define MAX_CLUSTERS 16

/** * CONSTANT MEMORY (Level 2 Speed)
 * Centroids are read by every thread in a warp simultaneously. 
 * The Constant Cache makes this much faster than Global memory.
 */
__constant__ float c_centroids_x[MAX_CLUSTERS];
__constant__ float c_centroids_y[MAX_CLUSTERS];

/**
 * GPU KERNEL: The Memory Waterfall
 * Logic: Global -> Shared -> Registers -> Global
 */
__global__ void kmeans_gpu_kernel(float *d_x, float *d_y, int *d_cluster, 
                                   int n_points, int k) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // SHARED MEMORY (Level 1 Speed - Block Wide)
    // We stage data here to reduce Global Memory pressure.
    extern __shared__ float s_data[];
    float *s_x = &s_data[0];
    float *s_y = &s_data[blockDim.x];

    if (tid < n_points) {
        // STEP 1: Move from Global to Shared
        s_x[local_id] = d_x[tid];
        s_y[local_id] = d_y[tid];
    }
    
    // Wait for all threads in the block to finish the Global -> Shared move
    __syncthreads(); 

    if (tid < n_points) {
        // STEP 2: Move from Shared to REGISTERS (Fastest Speed - Thread Private)
        // High-frequency math is performed only on register variables.
        float px = s_x[local_id]; 
        float py = s_y[local_id]; 
        
        // These tracking variables live in Registers
        float min_dist = FLT_MAX;
        int best_cluster = -1;

        // STEP 3: The Calculation Phase
        // Math is done in Registers. Centroid data comes from Constant Cache.
        for (int i = 0; i < k; i++) {
            // These temporary variables are pulled into registers for math
            float cx = c_centroids_x[i]; 
            float cy = c_centroids_y[i]; 
            
            float dx = px - cx;
            float dy = py - cy;
            float dist = (dx * dx) + (dy * dy); 

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = i;
            }
        }

        // STEP 4: Move result back to Global Memory
        // Once calculations are complete, move the final result out of registers.
        d_cluster[tid] = best_cluster;
    }
}

// HOST (CPU) VERSION: For performance baseline comparison
void kmeans_cpu(float *x, float *y, int *cluster, float *cx, float *cy, 
                int n, int k) {
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
    int n_points = blocks * threads;
    int k = 8;
    size_t f_size = n_points * sizeof(float);
    size_t i_size = n_points * sizeof(int);

    // HOST MEMORY: Standard CPU-side allocation
    float *h_x = (float*)malloc(f_size);
    float *h_y = (float*)malloc(f_size);
    int *h_cluster_gpu = (int*)malloc(i_size);
    int *h_cluster_cpu = (int*)malloc(i_size);
    float h_cx[MAX_CLUSTERS], h_cy[MAX_CLUSTERS];

    for(int i=0; i<n_points; i++) { h_x[i] = (float)(rand()%100); h_y[i] = (float)(rand()%100); }
    for(int i=0; i<k; i++) { h_cx[i] = (float)(rand()%100); h_cy[i] = (float)(rand()%100); }

    // --- CPU TIMING (HOST) ---
    clock_t cpu_start = clock();
    kmeans_cpu(h_x, h_y, h_cluster_cpu, h_cx, h_cy, n_points, k);
    clock_t cpu_stop = clock();
    double cpu_ms = ((double)(cpu_stop - cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // --- GPU TIMING (DEVICE) ---
    float *d_x, *d_y;
    int *d_cluster;
    cudaMalloc(&d_x, f_size); // GLOBAL MEMORY allocation
    cudaMalloc(&d_y, f_size);
    cudaMalloc(&d_cluster, i_size);

    cudaMemcpy(d_x, h_x, f_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, f_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_centroids_x, h_cx, k * sizeof(float)); // CONSTANT move
    cudaMemcpyToSymbol(c_centroids_y, h_cy, k * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t shared_mem_size = threads * sizeof(float) * 2; 
    kmeans_gpu_kernel<<<blocks, threads, shared_mem_size>>>(d_x, d_y, d_cluster, n_points, k);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Calculate total threads
    int total_threads = blocks * threads;

    printf("\n==============================================\n");
    printf("TEST CONFIGURATION: %d Blocks | %d Threads/Block\n", blocks, threads);
    printf("TOTAL PARALLEL THREADS: %d\n", total_threads);
    printf("MEMORY ENGAGED: Host, Global, Constant, Shared, Registers\n");
    printf("----------------------------------------------\n");
    printf("Host (CPU) Time:   %10.4f ms\n", cpu_ms);
    printf("Device (GPU) Time: %10.4f ms\n", gpu_ms);
    printf("Speedup Ratio:     %10.2fx\n", cpu_ms / gpu_ms);
    
    // Anecdotal check for the user
    if (gpu_ms < cpu_ms) {
        printf("RESULT: GPU is outperforming CPU by %0.2fms\n", cpu_ms - gpu_ms);
    } else {
        printf("RESULT: CPU is faster due to Kernel Launch Overhead\n");
    }
    printf("==============================================\n");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_cluster);
    free(h_x); free(h_y); free(h_cluster_gpu); free(h_cluster_cpu);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./assignment <blocks> <threads>\n");
        return 1;
    }
    run_benchmark(atoi(argv[1]), atoi(argv[2]));
    return 0;
}
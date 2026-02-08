#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel that performs a simple branch-free transformation on each element.
 * 
 * For each index `idx`, calculates y[idx] = x[idx] * 2 + 3.
 * Bounds checking is included to avoid out-of-bounds memory access.
 *
 * @param x Input array on device
 * @param y Output array on device
 * @param N Total number of elements to process
 */
__global__ void transform_kernel(int *x, int *y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds checking - not considered branchy
	if(idx < N)
        y[idx] = x[idx] * 2 + 3;
}


/**
 * @brief CUDA kernel that performs a branched transformation on each element.
 * 
 * For each index `idx`, checks if x[idx] is even or odd:
 * - If even: y[idx] = x[idx] * 2
 * - If odd:  y[idx] = x[idx] * 3
 * Bounds checking is included to prevent out-of-bounds access.
 *
 * @param x Input array on device
 * @param y Output array on device
 * @param N Total number of elements to process
 */
__global__ void transform_branch_kernel(int *x, int *y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds checking - not the actual branching
	if(idx < N)
    {
		// the branching logic on odd/even idx
        if(x[idx] % 2 == 0)
            y[idx] = x[idx] * 2;
        else
            y[idx] = x[idx] * 3;
    }
}


/**
 * @brief Main function: sets up host and device memory, runs kernels, and measures execution time.
 * 
 * Supports command-line arguments to vary total threads and threads per block:
 * ./assignment.exe [totalThreads] [blockSize]
 *
 * Performs both branch-free and branchy transformations, and prints timing results.
 */
int main(int argc, char** argv)
{
    // Default arguments (makes it easy for testing)
    int totalThreads = 1 << 20;  // 1,048,576 elements
    int blockSize = 256;

	// Get user passed args if given
    if(argc >= 2) totalThreads = atoi(argv[1]);
    if(argc >= 3) blockSize = atoi(argv[2]);

	// Calc block size
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    printf("GPU: Total threads = %d, Block size = %d, Num blocks = %d\n", totalThreads, blockSize, numBlocks);

    // Allocate host arrays
    int *h_x = (int*)malloc(totalThreads * sizeof(int));
    int *h_y = (int*)malloc(totalThreads * sizeof(int));
    int *h_y_branch = (int*)malloc(totalThreads * sizeof(int));

	// Init host array
    for(int i=0; i<totalThreads; i++)
        h_x[i] = rand() % 100;

    // Allocate device arrays
    int *d_x, *d_y;
    cudaMalloc((void**)&d_x, totalThreads*sizeof(int));
    cudaMalloc((void**)&d_y, totalThreads*sizeof(int));

    cudaMemcpy(d_x, h_x, totalThreads*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);

    // Branch free test
    cudaEventRecord(startEvt);
    transform_kernel<<<numBlocks, blockSize>>>(d_x, d_y, totalThreads);
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, startEvt, stopEvt);
    printf("GPU branch-free time: %f ms\n", gpu_time);

    cudaMemcpy(h_y, d_y, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);

    // Branched test
    cudaEventRecord(startEvt);
    transform_branch_kernel<<<numBlocks, blockSize>>>(d_x, d_y, totalThreads);
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);

    float gpu_branch_time = 0;
    cudaEventElapsedTime(&gpu_branch_time, startEvt, stopEvt);
    printf("GPU branchy time: %f ms\n", gpu_branch_time);

    cudaMemcpy(h_y_branch, d_y, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_y_branch);

    return 0;
}

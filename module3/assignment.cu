#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Branch-free CUDA kernel
__global__ void transform_kernel(int *x, int *y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
        y[idx] = x[idx] * 2 + 3;
}

// Branchy CUDA kernel
__global__ void transform_branch_kernel(int *x, int *y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        if(x[idx] % 2 == 0)
            y[idx] = x[idx] * 2;
        else
            y[idx] = x[idx] * 3;
    }
}

int main(int argc, char** argv)
{
    // Default arguments
    int totalThreads = 1 << 20;
    int blockSize = 256;

    if(argc >= 2) totalThreads = atoi(argv[1]);
    if(argc >= 3) blockSize = atoi(argv[2]);

    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    printf("GPU: Total threads = %d, Block size = %d, Num blocks = %d\n",
           totalThreads, blockSize, numBlocks);

    // Allocate host arrays
    int *h_x = (int*)malloc(totalThreads * sizeof(int));
    int *h_y = (int*)malloc(totalThreads * sizeof(int));
    int *h_y_branch = (int*)malloc(totalThreads * sizeof(int));

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

    // --- GPU branch-free ---
    cudaEventRecord(startEvt);
    transform_kernel<<<numBlocks, blockSize>>>(d_x, d_y, totalThreads);
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, startEvt, stopEvt);
    printf("GPU branch-free time: %f ms\n", gpu_time);

    cudaMemcpy(h_y, d_y, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);

    // --- GPU branchy ---
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

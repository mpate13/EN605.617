#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

/**
 * --- CONSTANTS AND CONFIGURATION ---
 */
#define HASH_ROUNDS 100         
#define ENCRYPT_ROUNDS 50
#define MASTER_SEED 0xACE1
#define GOLDEN_RATIO_PRIME 0x9e3779b9
#define ROTATE_LEFT_BITS 3
#define ROTATE_RIGHT_BITS 29
#define HASH_SHIFT_LEFT 5
#define HASH_SHIFT_RIGHT 3
#define NUM_RUNS 2

/**
 * gpuErrchk / gpuAssert
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s at %s:%d\n", 
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// --- FORWARD DECLARATION OR DEFINITION FIRST ---
// Defining this before cpuEncrypt fixes the "undefined" error.

unsigned int cpuParallelKeyGen(unsigned int seed, int idx) {
    unsigned int val = seed + (unsigned int)idx;
    for (int i = 0; i < HASH_ROUNDS; i++) {
        // No magic numbers here!
        val = ((val << HASH_SHIFT_LEFT) ^ (val >> HASH_SHIFT_RIGHT)) + GOLDEN_RATIO_PRIME;
    }
    return val;
}

// --- CUDA KERNELS ---

__global__ void kernelParallelKeyGenerator(unsigned int *deviceKeyBuffer, 
                                           unsigned int seed, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int val = seed + idx;
        for (int i = 0; i < HASH_ROUNDS; i++) {
            // Using constants in kernel as well
            val = ((val << HASH_SHIFT_LEFT) ^ (val >> HASH_SHIFT_RIGHT)) + GOLDEN_RATIO_PRIME;
        }
        deviceKeyBuffer[idx] = val; 
    }
}

__global__ void kernelEncryptData(unsigned int *dataBuffer, 
                                  unsigned int *keyBuffer, int numElements) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < numElements) {
        unsigned int key = keyBuffer[globalIdx];
        unsigned int val = dataBuffer[globalIdx];
        for(int i = 0; i < ENCRYPT_ROUNDS; i++) {
            val ^= key;
            val = (val << ROTATE_LEFT_BITS) | (val >> ROTATE_RIGHT_BITS); 
        }
        dataBuffer[globalIdx] = val;
    }
}

// --- RESOURCE MANAGEMENT ---

void initCudaResources(cudaStream_t *sKey, cudaStream_t *sEnc, 
                       cudaEvent_t *start, cudaEvent_t *stop, cudaEvent_t *ready) {
    gpuErrchk(cudaStreamCreate(sKey));
    gpuErrchk(cudaStreamCreate(sEnc));
    gpuErrchk(cudaEventCreate(start));
    gpuErrchk(cudaEventCreate(stop));
    gpuErrchk(cudaEventCreate(ready));
}

void cleanupCudaResources(cudaStream_t s1, cudaStream_t s2, 
                          cudaEvent_t e1, cudaEvent_t e2, cudaEvent_t e3) {
    cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
}

void runGpuPipeline(unsigned int *deviceData, unsigned int *hostData, 
                    unsigned int *deviceKeyBuffer, size_t bufferSize, 
                    int numElements, int numBlocks, int threadsPerBlock, 
                    float *elapsedMs) {
    
    cudaStream_t streamKey, streamEncrypt;
    cudaEvent_t startEvent, stopEvent, keyReadyEvent;

    initCudaResources(&streamKey, &streamEncrypt, &startEvent, &stopEvent, &keyReadyEvent);
    
    gpuErrchk(cudaEventRecord(startEvent, streamKey));

    kernelParallelKeyGenerator<<<numBlocks, threadsPerBlock, 0, streamKey>>>(
        deviceKeyBuffer, MASTER_SEED, numElements);
    gpuErrchk(cudaEventRecord(keyReadyEvent, streamKey));

    gpuErrchk(cudaMemcpyAsync(deviceData, hostData, bufferSize, 
                              cudaMemcpyHostToDevice, streamEncrypt));

    gpuErrchk(cudaStreamWaitEvent(streamEncrypt, keyReadyEvent, 0));
    
    kernelEncryptData<<<numBlocks, threadsPerBlock, 0, streamEncrypt>>>(
        deviceData, deviceKeyBuffer, numElements);

    gpuErrchk(cudaMemcpyAsync(hostData, deviceData, bufferSize, 
                              cudaMemcpyDeviceToHost, streamEncrypt));

    gpuErrchk(cudaEventRecord(stopEvent, streamEncrypt));
    gpuErrchk(cudaEventSynchronize(stopEvent)); 
    gpuErrchk(cudaEventElapsedTime(elapsedMs, startEvent, stopEvent));

    cleanupCudaResources(streamKey, streamEncrypt, startEvent, stopEvent, keyReadyEvent);
}

// --- CPU REFERENCE CODE ---
void cpuEncrypt(unsigned int *data, int n) {
    for (int i = 0; i < n; i++) {
        unsigned int key = cpuParallelKeyGen(MASTER_SEED, i);
        unsigned int val = data[i];
        for(int j = 0; j < ENCRYPT_ROUNDS; j++) {
            val ^= key;
            val = (val << ROTATE_LEFT_BITS) | (val >> ROTATE_RIGHT_BITS); 
        }
        data[i] = val;
    }
}

// --- TEST HARNESS ---

void run_test_iteration(unsigned int *hostData, unsigned int *deviceData, 
                        unsigned int *cpuRef, unsigned int *deviceKeyBuffer, 
                        size_t bufferSize, int numElements, int numBlocks, 
                        int threadsPerBlock, int runId) {
    
    for (int i = 0; i < numElements; i++) {
        hostData[i] = (unsigned int)i;
        cpuRef[i] = (unsigned int)i;
    }

    double cpuStart = get_current_time_ms();
    cpuEncrypt(cpuRef, numElements); 
    double cpuElapsedMs = get_current_time_ms() - cpuStart;

    float gpuElapsedMs = 0;
    runGpuPipeline(deviceData, hostData, deviceKeyBuffer, bufferSize, 
                   numElements, numBlocks, threadsPerBlock, &gpuElapsedMs);
    
    int errorCount = 0;
    for (int i = 0; i < numElements; i++) {
        if (hostData[i] != cpuRef[i]) errorCount++;
    }
    
    printf("[Run %d] TotalThreads: %d | NumBlocks: %d | BlockSize: %d\n", 
           runId, numElements, numBlocks, threadsPerBlock);
    printf("        Status: %s (%d errors) | CPU: %.2fms | GPU: %.2fms\n\n", 
           (errorCount == 0) ? "PASS" : "FAIL", errorCount, cpuElapsedMs, gpuElapsedMs);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <total_elements> <block_size>\n", argv[0]);
        return 1;
    }
    
    int totalElements = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    size_t bufferSize = totalElements * sizeof(unsigned int);

    unsigned int *hostDataPinned, *deviceData, *deviceKeyBuffer, *cpuRef;
    cpuRef = (unsigned int*)malloc(bufferSize);
    
    gpuErrchk(cudaHostAlloc(&hostDataPinned, bufferSize, 0));
    gpuErrchk(cudaMalloc(&deviceData, bufferSize));
    gpuErrchk(cudaMalloc(&deviceKeyBuffer, bufferSize)); 

    for(int i = 0; i < NUM_RUNS; i++) {
        run_test_iteration(hostDataPinned, deviceData, cpuRef, 
                           deviceKeyBuffer, bufferSize, totalElements, 
                           numBlocks, threadsPerBlock, i);
    }

    gpuErrchk(cudaFreeHost(hostDataPinned)); 
    gpuErrchk(cudaFree(deviceData));
    gpuErrchk(cudaFree(deviceKeyBuffer)); 
    free(cpuRef);
    
    return 0;
}
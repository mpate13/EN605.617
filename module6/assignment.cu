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
 * @brief Error handling macro for CUDA API calls.
 * Prints the error string, file name, and line number before exiting.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s at %s:%d\n", 
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/**
 * @brief Retrieves the current wall-clock time in milliseconds.
 * Useful for calculating CPU execution time and high-level benchmarking.
 * @return Current time as a double in milliseconds.
 */
double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

/**
 * @brief CPU-based key generation logic for a single element.
 * Applies a series of bitwise shifts and XORs using the Golden Ratio Prime 
 * to transform a seed and index into a pseudo-random key.
 * @param seed The base master seed.
 * @param idx  The unique index for this specific data element.
 * @return The generated 32-bit unsigned integer key.
 */
unsigned int cpuParallelKeyGen(unsigned int seed, int idx) {
    unsigned int val = seed + (unsigned int)idx;
    for (int i = 0; i < HASH_ROUNDS; i++) {
        val = ((val << HASH_SHIFT_LEFT) ^ (val >> HASH_SHIFT_RIGHT)) + 
        GOLDEN_RATIO_PRIME;
    }
    return val;
}

// --- CUDA KERNELS ---

/**
 * @brief GPU Kernel to generate encryption keys in parallel.
 * Each CUDA thread calculates its own key based on its global thread index.
 * @param deviceKeyBuffer Pointer to GPU memory where keys will be stored.
 * @param seed            The master seed used for the hash.
 * @param numElements     The total number of keys to generate.
 */
__global__ void kernelParallelKeyGenerator(unsigned int *deviceKeyBuffer, 
                                           unsigned int seed, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int val = seed + idx;
        for (int i = 0; i < HASH_ROUNDS; i++) {
            val = ((val << HASH_SHIFT_LEFT) ^ (val >> HASH_SHIFT_RIGHT)) + 
            GOLDEN_RATIO_PRIME;
        }
        deviceKeyBuffer[idx] = val; 
    }
}

/**
 * @brief GPU Kernel to encrypt data using generated keys.
 * Performs ENCRYPT_ROUNDS of XOR and bitwise rotation on each element.
 * @param dataBuffer  Pointer to GPU memory containing the data to encrypt.
 * @param keyBuffer   Pointer to GPU memory containing the generated keys.
 * @param numElements The total number of elements to process.
 */
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

/**
 * Allocates and initializes CUDA streams and events.
 * Used to set up the infrastructure for concurrent execution and 
 * synchronization.
 * @param sKey  Pointer to the stream for key generation.
 * @param sEnc  Pointer to the stream for data transfer and encryption.
 * @param start Event to mark the beginning of the pipeline.
 * @param stop  Event to mark the end of the pipeline.
 * @param ready Event to signal that keys are ready for use by the 
 * encryption stream.
 */
void initCudaResources(cudaStream_t *sKey, cudaStream_t *sEnc, 
                       cudaEvent_t *start, cudaEvent_t *stop, 
                       cudaEvent_t *ready) {
    gpuErrchk(cudaStreamCreate(sKey));
    gpuErrchk(cudaStreamCreate(sEnc));
    gpuErrchk(cudaEventCreate(start));
    gpuErrchk(cudaEventCreate(stop));
    gpuErrchk(cudaEventCreate(ready));
}

/**
 * Destroys CUDA streams and events to free GPU resources.
 */
void cleanupCudaResources(cudaStream_t s1, cudaStream_t s2, 
                          cudaEvent_t e1, cudaEvent_t e2, cudaEvent_t e3) {
    cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
}

/**
 * GPU execution pipeline using multiple streams.
 * Coordinates key generation, asynchronous memory transfers, and kernel 
 * execution while using events to ensure dependencies are met.
 * @param deviceData      Pointer to GPU data buffer.
 * @param hostData        Pointer to CPU data buffer (pinned memory).
 * @param deviceKeyBuffer Pointer to GPU key buffer.
 * @param bufferSize      Total size of memory in bytes.
 * @param numElements     Total count of integers.
 * @param numBlocks       Number of CUDA blocks.
 * @param threadsPerBlock Number of threads per CUDA block.
 * @param elapsedMs       Output pointer for the measured GPU execution time.
 */
void runGpuPipeline(unsigned int *deviceData, unsigned int *hostData, 
                    unsigned int *deviceKeyBuffer, size_t bufferSize, 
                    int numElements, int numBlocks, int threadsPerBlock, 
                    float *elapsedMs) {
    
    cudaStream_t streamKey, streamEncrypt;
    cudaEvent_t startEvent, stopEvent, keyReadyEvent;

    initCudaResources(&streamKey, &streamEncrypt, &startEvent, &stopEvent, &keyReadyEvent);
    
    // Start timing
    gpuErrchk(cudaEventRecord(startEvent, streamKey));

    // Generate keys in Stream A
    kernelParallelKeyGenerator<<<numBlocks, threadsPerBlock, 0, streamKey>>>(
        deviceKeyBuffer, MASTER_SEED, numElements);
    gpuErrchk(cudaEventRecord(keyReadyEvent, streamKey));

    // Transfer data from Host to Device in Stream B
    gpuErrchk(cudaMemcpyAsync(deviceData, hostData, bufferSize, 
                              cudaMemcpyHostToDevice, streamEncrypt));

    // Stream B waits for Stream A's key generation to finish
    gpuErrchk(cudaStreamWaitEvent(streamEncrypt, keyReadyEvent, 0));
    
    // Encrypt data in Stream B
    kernelEncryptData<<<numBlocks, threadsPerBlock, 0, streamEncrypt>>>(
        deviceData, deviceKeyBuffer, numElements);

    // Transfer encrypted data back to Host in Stream B
    gpuErrchk(cudaMemcpyAsync(hostData, deviceData, bufferSize, 
                              cudaMemcpyDeviceToHost, streamEncrypt));

    // Stop timing
    gpuErrchk(cudaEventRecord(stopEvent, streamEncrypt));
    gpuErrchk(cudaEventSynchronize(stopEvent)); 
    gpuErrchk(cudaEventElapsedTime(elapsedMs, startEvent, stopEvent));

    cleanupCudaResources(streamKey, streamEncrypt, startEvent, stopEvent, 
        keyReadyEvent);
}

// CPU VERSION FOR TESTING

/**
 * Serial CPU implementation of the encryption logic.
 * Used as a reference to verify the correctness of the GPU results.
 * @param data Array of data to be encrypted in-place.
 * @param n    Number of elements in the array.
 */
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

/**
 * Initializes data, runs both CPU and GPU versions, and compares 
 * results.
 * Outputs the performance metrics and a PASS/FAIL status for the iteration.
 * @param runId Index of the current test iteration.
 */
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
           (errorCount == 0) ? "PASS" : "FAIL", errorCount, cpuElapsedMs, 
           gpuElapsedMs);
}

/**
 * Entry point for the program.
 * Parses command line arguments, allocates memory, and manages the test loop.
 */
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
    
    // Allocate Pinned Host Memory and Device Memory
    gpuErrchk(cudaHostAlloc(&hostDataPinned, bufferSize, 0));
    gpuErrchk(cudaMalloc(&deviceData, bufferSize));
    gpuErrchk(cudaMalloc(&deviceKeyBuffer, bufferSize)); 

    for(int i = 0; i < NUM_RUNS; i++) {
        run_test_iteration(hostDataPinned, deviceData, cpuRef, 
                           deviceKeyBuffer, bufferSize, totalElements, 
                           numBlocks, threadsPerBlock, i);
    }

    // Cleanup
    gpuErrchk(cudaFreeHost(hostDataPinned)); 
    gpuErrchk(cudaFree(deviceData));
    gpuErrchk(cudaFree(deviceKeyBuffer)); 
    free(cpuRef);
    
    return 0;
}
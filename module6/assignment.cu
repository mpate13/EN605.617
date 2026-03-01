#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// --- CONSTANTS ---
#define HASH_ROUNDS 500  
#define ENCRYPT_ROUNDS 50
#define MASTER_SEED 0xACE1
#define GOLDEN_RATIO_PRIME 0x9e3779b9
#define ROTATE_LEFT_BITS 3
#define ROTATE_RIGHT_BITS 29
#define HASH_SHIFT_LEFT  5
#define HASH_SHIFT_RIGHT 3

// --- 1. CONSTANT MEMORY ---
__constant__ unsigned int dc_key;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s at %s:%d\n", 
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

double get_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// kernel key producer
__global__ void producerKeyGen(unsigned int *d_temp_key, unsigned int seed) {
    unsigned int val = seed;
    for (int i = 0; i < HASH_ROUNDS; i++) {
        unsigned int mixed_bits = (val << HASH_SHIFT_LEFT) ^ 
        (val >> HASH_SHIFT_RIGHT);
        val = mixed_bits + GOLDEN_RATIO_PRIME;
    }
    *d_temp_key = val; 
}

__global__ void consumerEncrypt(unsigned int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int key = dc_key; 
        unsigned int val = data[tid];
        for(int i = 0; i < ENCRYPT_ROUNDS; i++) {
            val ^= key;
            val = (val << ROTATE_LEFT_BITS) | (val >> ROTATE_RIGHT_BITS); 
        }
        data[tid] = val;
    }
}

// --- 3. PIPELINE WITH CROSS-STREAM SYNC ---
void runGpuPipeline(unsigned int *d_d, unsigned int *h_d, 
                    unsigned int *d_temp_k, size_t sz, int n, int b, 
                    int t, float *ms) {
    
    cudaStream_t streamKey, streamEncrypt;
    cudaEvent_t start, stop, keyReady;
    unsigned int h_key;

    // Initialize Streams and Events
    gpuErrchk(cudaStreamCreate(&streamKey));
    gpuErrchk(cudaStreamCreate(&streamEncrypt));
    gpuErrchk(cudaEventCreate(&start)); 
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventCreate(&keyReady)); // Used to signal between streams

    // Record start time using the Key stream
    gpuErrchk(cudaEventRecord(start, streamKey));
    
    // --- WORKFLOW 1: KEY GENERATION (Stream A) ---
    producerKeyGen<<<1, 1, 0, streamKey>>>(d_temp_k, MASTER_SEED);
    
    // Record 'keyReady' event immediately after 
    // the kernel is queued in streamKey
    gpuErrchk(cudaEventRecord(keyReady, streamKey));

    // To use the key in Constant Memory (dc_key), we must pull it to host first
    gpuErrchk(cudaMemcpyAsync(&h_key, d_temp_k, sizeof(unsigned int), 
    cudaMemcpyDeviceToHost, streamKey));
    gpuErrchk(cudaStreamSynchronize(streamKey)); 
    gpuErrchk(cudaMemcpyToSymbol(dc_key, &h_key, sizeof(unsigned int)));

    // --- WORKFLOW 2: ENCRYPTION (Stream B) ---
    // Start data transfer in Stream B 
    // (Can overlap with Key Generation in Stream A)
    gpuErrchk(cudaMemcpyAsync(d_d, h_d, sz, cudaMemcpyHostToDevice, 
        streamEncrypt));

    // CROSS-STREAM SYNC: Tell streamEncrypt to wait for keyReady 
    // event from streamKey
    // This allows the transfer to happen, but the kernel waits for the signal
    gpuErrchk(cudaStreamWaitEvent(streamEncrypt, keyReady, 0));

    consumerEncrypt<<<b, t, 0, streamEncrypt>>>(d_d, n);
    
    // Move results back to host
    gpuErrchk(cudaMemcpyAsync(h_d, d_d, sz, cudaMemcpyDeviceToHost, 
        streamEncrypt));

    // Record stop on the final stream and sync
    gpuErrchk(cudaEventRecord(stop, streamEncrypt));
    gpuErrchk(cudaEventSynchronize(stop)); 
    
    gpuErrchk(cudaEventElapsedTime(ms, start, stop));

    // Cleanup
    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
    cudaEventDestroy(keyReady);
    cudaStreamDestroy(streamKey); 
    cudaStreamDestroy(streamEncrypt);
}

// --- 4. CPU LOGIC ---
unsigned int cpuKeyGen(unsigned int seed) {
    unsigned int val = seed;
    for (int i = 0; i < HASH_ROUNDS; i++) 
        val = ((val << 5) ^ (val >> 3)) + GOLDEN_RATIO_PRIME;
    return val;
}

void cpuEncrypt(unsigned int *data, unsigned int key, int n) {
    for (int i = 0; i < n; i++) {
        unsigned int val = data[i];
        for(int j = 0; j < ENCRYPT_ROUNDS; j++) {
            val ^= key;
            val = (val << ROTATE_LEFT_BITS) | (val >> ROTATE_RIGHT_BITS); 
        }
        data[i] = val;
    }
}

// --- 5. TEST HARNESS ---
void run_test(unsigned int *h_d, unsigned int *d_d, unsigned int *cpu_d, 
              unsigned int *d_k, size_t sz, int n, int b, int t, int run_id) {
    
    // Reset buffers
    for (int i = 0; i < n; i++) h_d[i] = cpu_d[i] = (unsigned int)i;

    double c_s = get_ms();
    cpuEncrypt(cpu_d, cpuKeyGen(MASTER_SEED), n); 
    double cpu_ms = get_ms() - c_s;

    float gpu_ms = 0;
    runGpuPipeline(d_d, h_d, d_k, sz, n, b, t, &gpu_ms);
    
    int errs = 0;
    for (int i = 0; i < n; i++) if (h_d[i] != cpu_d[i]) errs++;
    
    printf("[Run %d] Size: %d | BlockSize: %d | GridSize: %d\n", 
        run_id, n, t, b);
    printf("        Status: %s (%d errors) | CPU: %.2fms | GPU: %.2fms\n", 
           (errs == 0) ? "PASS" : "FAIL", errs, cpu_ms, gpu_ms);
    printf("-------------------------
        -----------------------------------------\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <total_threads> <block_size>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int bs = atoi(argv[2]);
    int blocks = (n + bs - 1) / bs;
    size_t sz = n * sizeof(unsigned int);

    unsigned int *h_data, *d_data, *d_key, *cpu_data;
    cpu_data = (unsigned int*)malloc(sz);
    
    // Use pinned memory for asynchronous transfer efficiency
    gpuErrchk(cudaHostAlloc(&h_data, sz, 0));
    gpuErrchk(cudaMalloc(&d_data, sz));
    gpuErrchk(cudaMalloc(&d_key, sizeof(unsigned int)));

    printf("Executing CUDA Streams/Events Program 
        (Multi-Stream Dependency Logic)\n\n");

    // Execute 2 separate runs as required by the rubric
    for(int i = 1; i <= 2; i++) {
        run_test(h_data, d_data, cpu_data, d_key, sz, n, blocks, bs, i);
    }

    gpuErrchk(cudaFreeHost(h_data)); 
    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_key)); 
    free(cpu_data);
    
    return 0;
}
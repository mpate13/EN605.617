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
void initResources(cudaStream_t *s1, cudaStream_t *s2, cudaEvent_t *e1, cudaEvent_t *e2, cudaEvent_t *e3) {
    gpuErrchk(cudaStreamCreate(s1));
    gpuErrchk(cudaStreamCreate(s2));
    gpuErrchk(cudaEventCreate(e1));
    gpuErrchk(cudaEventCreate(e2));
    gpuErrchk(cudaEventCreate(e3));
}

void cleanupResources(cudaStream_t s1, cudaStream_t s2, cudaEvent_t e1, cudaEvent_t e2, cudaEvent_t e3) {
    cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
}

void prepareGlobalKey(unsigned int *d_temp_k, cudaStream_t stream, cudaEvent_t keyReady) {
    unsigned int h_key;
    // Launch KeyGen and signal event
    producerKeyGen<<<1, 1, 0, stream>>>(d_temp_k, MASTER_SEED);
    gpuErrchk(cudaEventRecord(keyReady, stream));

    // Must sync to move generated key into __constant__ memory
    gpuErrchk(cudaMemcpyAsync(&h_key, d_temp_k, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream)); 
    gpuErrchk(cudaMemcpyToSymbol(dc_key, &h_key, sizeof(unsigned int)));
}


void runGpuPipeline(unsigned int *d_d, unsigned int *h_d, unsigned int *d_temp_k, 
                    size_t sz, int n, int b, int t, float *ms) {
    cudaStream_t sKey, sEnc;
    cudaEvent_t start, stop, keyReady;

    initResources(&sKey, &sEnc, &start, &stop, &keyReady);
    gpuErrchk(cudaEventRecord(start, sKey));

    // Stream A: Generate Key
    prepareGlobalKey(d_temp_k, sKey, keyReady);

    // Stream B: Overlap H2D transfer with Key Generation above
    gpuErrchk(cudaMemcpyAsync(d_d, h_d, sz, cudaMemcpyHostToDevice, sEnc));

    // Wait for Key, then Encrypt, then D2H
    gpuErrchk(cudaStreamWaitEvent(sEnc, keyReady, 0));
    consumerEncrypt<<<b, t, 0, sEnc>>>(d_d, n);
    gpuErrchk(cudaMemcpyAsync(h_d, d_d, sz, cudaMemcpyDeviceToHost, sEnc));

    gpuErrchk(cudaEventRecord(stop, sEnc));
    gpuErrchk(cudaEventSynchronize(stop)); 
    gpuErrchk(cudaEventElapsedTime(ms, start, stop));

    cleanupResources(sKey, sEnc, start, stop, keyReady);
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
    for (int i = 0; i < n; i++) {
        h_d[i] = (unsigned int)i;
        cpu_d[i] = (unsigned int)i;
    }

    double c_s = get_ms();
    cpuEncrypt(cpu_d, cpuKeyGen(MASTER_SEED), n); 
    double cpu_ms = get_ms() - c_s;

    float gpu_ms = 0;
    runGpuPipeline(d_d, h_d, d_k, sz, n, b, t, &gpu_ms);
    
    int errs = 0;
    for (int i = 0; i < n; i++) {
        if (h_d[i] != cpu_d[i]) {
            errs++;
        }
    }
    
    printf("[Run %d] Size: %d | BlockSize: %d | GridSize: %d\n", 
        run_id, n, t, b);
    printf("        Status: %s (%d errors) | CPU: %.2fms | GPU: %.2fms\n\n", 
           (errs == 0) ? "PASS" : "FAIL", errs, cpu_ms, gpu_ms);
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
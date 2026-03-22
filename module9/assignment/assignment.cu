/*
 * ----------------------------------------------------------------------------
 * MOTIVATION: Solving the "Scan Time" Problem in MRI
 * ----------------------------------------------------------------------------
 * Problem: Traditional MRI is slow because it requires a full sampling of 
 * "k-space" which is a freqency map of the body. Long scan times lead to 
 * patient movement  and high costs.
 *
 * NOTE: the issue isn't a compute time issue, its a MRI machine issue
 * If a patient moves even a millimeter during those 20 minutes, the entire 
 * k-space dataset is corrupted, and the image comes out as a blurry mess.
 * By only keeping 10-20% data, it takes 2–4 minutes. 
 * This is short enough for a patient to hold their breath or for a 
 * child to stay still.
 *
 * Solution: This project implements Compressed Sensing logic. By treating 
 * k-space as a Sparse Matrix, we can undersample the frequencies (collecting 
 * only 10-20% of the data).
 * * Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC4984938/
 * ----------------------------------------------------------------------------
 * EXECUTION FLOW: What is happening?
 * ----------------------------------------------------------------------------
 * 1. IMAGE UPLOAD:  8-bit grayscale pixels are normalized [0,1] and 
 * transferred to GPU VRAM as Complex (float2) values.
 * 2. cuFFT FORWARD: Transforms the spatial image into the frequency domain 
 * (k-space). Each pixel becomes a sum of sine waves.
 * 3. cuSPARSE MASK: Applies a Sparse Sampling Matrix (CSR format) to the 
 * k-space data. This zeros out 90% of the frequencies, 
 * simulating a high-speed, undersampled scan.
 * 4. cuFFT INVERSE: Reconstructs the organ image from the sparse frequency 
 * data. 
 * 5. DOWNLOAD: Final image is normalized, converted back to 8-bit, 
 * and saved as a reconstructed JPEG.
 * ----------------------------------------------------------------------------
 *
 * NOTE ON cuSPARSE: My current implementation uses the CSR format 
 * currently uses a simple diagonal mask, but in a future version, 
 * this function can easily be updated to work with  
 * non-Cartesian (Radial/Spiral) trajectories (for the sake of the
 * assignmnet, wanted to make the program work vs learning a more complex
 * math technique).
 *
 * By keeping all math in VRAM and using CUDA Streams, we maximize throughput 
 * and minimize host-to-device overhead.
 * ----------------------------------------------------------------------------
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;
typedef float2 Complex;

// Local error checking to replace missing helper_cuda.h
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/**
 * Performs Sparse Matrix-Vector Multiplication (SpMV) using the cuSPARSE 
 * Generic API on a specific stream. This acts as a digital filter, zeroing 
 * out unsampled frequencies in the k-space signal.
 * https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-apis
 */
void executeSparseKernelMask(cusparseHandle_t handle, cudaStream_t stream,
                             int n, int *d_row, int *d_col, float *d_val, 
                             float *d_in, float *d_out) {
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f, beta = 0.0f;

    // Fixed USPARSE typo to CUSPARSE
    cusparseCreateCsr(&matA, n, n, n, d_row, d_col, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, n, d_in, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, d_out, CUDA_R_32F);

    cusparseSpMV_bufferSize(handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, 
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    checkCuda(cudaMalloc((void**)&dBuffer, bufferSize));
    
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, 
                 matA, vecX, &beta, vecY, CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    checkCuda(cudaFree(dBuffer));
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
}

/**
 * Initializes the undersampling mask in CSR format. 
 * NOTE: While this implementation uses a diagonal structure for Cartesian 
 * validation, (which might be a more basic operation), this framework 
 * can easily support more complex math operations
 * non-Cartesian (Radial/Spiral) gridding matrices where off-diagonal 
 * sparsity patterns are required for frequency interpolation.
 * * NOTE: For the sake of time, kept it simple because didn't want to dig 
 * too much into the math
 */
void initializeUndersamplingMask(int n, int **d_row, int **d_col, 
                                 float **d_val) {
    int *h_row = (int*)malloc((n + 1) * sizeof(int));
    int *h_col = (int*)malloc(n * sizeof(int));
    float *h_val = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_row[i] = i;
        h_col[i] = i;
        h_val[i] = (i < n / 10) ? 1.0f : 0.0f;
    }
    h_row[n] = n;

    checkCuda(cudaMalloc((void**)d_row, (n + 1) * sizeof(int)));
    checkCuda(cudaMalloc((void**)d_col, n * sizeof(int)));
    checkCuda(cudaMalloc((void**)d_val, n * sizeof(float)));

    // Replaced shorthand H2D with standard constant
    checkCuda(cudaMemcpy(*d_row, h_row, (n + 1) * sizeof(int), 
    cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(*d_col, h_col, n * sizeof(int), 
    cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(*d_val, h_val, n * sizeof(float), 
    cudaMemcpyHostToDevice));

    free(h_row); free(h_col); free(h_val);
}

/**
 * Prepares image data for complex-to-complex transformations. Normalizes 
 * pixel values to [0,1] and allocates global memory buffers on the GPU.
 */
void uploadImageDataToGPU(unsigned char *img, int n, Complex **d_kSpace, 
                          Complex **d_masked) {
    Complex *h_temp = new Complex[n];
    for (int i = 0; i < n; i++) { 
        h_temp[i].x = (float)img[i] / 255.0f; 
        h_temp[i].y = 0.0f; 
    }

    checkCuda(cudaMalloc((void**)d_kSpace, n * sizeof(Complex)));
    checkCuda(cudaMalloc((void**)d_masked, n * sizeof(Complex)));
    checkCuda(cudaMemcpy(*d_kSpace, h_temp, n * sizeof(Complex), 
    cudaMemcpyHostToDevice));
    
    delete[] h_temp;
}

/**
 * Performs the forward FFT to reach k-space and applies the sparse
 * sampling mask using the cuSPARSE SpMV engine.
 */
void processFrequencyDomain(cufftHandle plan, cusparseHandle_t handle, 
                            cudaStream_t stream, int totalPixels, 
                            Complex *d_kSpace, Complex *d_masked, 
                            int *d_rows, int *d_cols, float *d_vals) {
    // Forward Transform to k-space
    cufftExecC2C(plan, (cufftComplex*)d_kSpace, 
                 (cufftComplex*)d_kSpace, CUFFT_FORWARD);

    // Apply Sparse Mask
    executeSparseKernelMask(handle, stream, totalPixels, d_rows, d_cols, 
                            d_vals, (float*)d_kSpace, (float*)d_masked);
}

/**
 * Reconstruction and Resource Management
 * Transforms data back to the spatial domain and synchronizes the stream
 * before releasing library handles.
 */
void reconstructAndCleanup(cufftHandle plan, cusparseHandle_t handle, 
                           cudaStream_t stream, Complex *d_masked) {
    // Inverse Transform to Spatial Image
    cufftExecC2C(plan, (cufftComplex*)d_masked, 
                 (cufftComplex*)d_masked, CUFFT_INVERSE);

    // Block until stream completes to ensure data integrity
    cudaStreamSynchronize(stream);

    cufftDestroy(plan);
    cusparseDestroy(handle);
    cudaStreamDestroy(stream);
}

/**
 * Runs the full signal processing pipeline on a custom CUDA stream. 
 * Performs forward 2D-FFT, sparse masking, and inverse 2D-FFT to 
 * reconstruct the spatial image.
 */
void executeReconstructionPipeline(int w, int h, int totalPixels, 
                                   Complex *d_kSpace, Complex *d_masked, 
                                   int *d_rows, int *d_cols, float *d_vals) {
    cufftHandle fftPlan;
    cusparseHandle_t sparseHandle;
    cudaStream_t computeStream;

    // Initialization
    checkCuda(cudaStreamCreate(&computeStream));
    cufftPlan2d(&fftPlan, h, w, CUFFT_C2C);
    cufftSetStream(fftPlan, computeStream);
    cusparseCreate(&sparseHandle);
    cusparseSetStream(sparseHandle, computeStream);

    // Execution
    processFrequencyDomain(fftPlan, sparseHandle, computeStream, totalPixels,
                           d_kSpace, d_masked, d_rows, d_cols, d_vals);

    reconstructAndCleanup(fftPlan, sparseHandle, computeStream, d_masked);
}

/**
 * Retrieves reconstructed data from the GPU, normalizes by FFT size, and 
 * converts complex results back to 8-bit grayscale for image output.
 */
void downloadAndSaveImage(Complex *d_masked, int width, int height) {
    int total_pixels = width * height;
    Complex *h_out = new Complex[total_pixels];
    unsigned char *final_img = (unsigned char*)malloc(total_pixels);

    // Replaced shorthand D2H with standard constant
    checkCuda(cudaMemcpy(h_out, d_masked, total_pixels * sizeof(Complex), 
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < total_pixels; i++) {
        float val = (h_out[i].x / total_pixels) * 255.0f;
        final_img[i] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }

    stbi_write_jpg("reconstructed_mri.jpg", width, height, 1, final_img, 100);
    
    delete[] h_out;
    free(final_img);
}

/**
 * Entry point. Initializes host resources, triggers the optimized 
 * GPU pipeline, and manages final cleanup. 
 */
int main(int argc, char** argv) {
    if (argc < 2) { 
        cout << "Usage: " << argv[0] << " <image.jpg>" << endl; 
        return 1; 
    }
    
    int width, height, channels;
    unsigned char *raw_img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!raw_img) { cout << "Error loading image." << endl; return 1; }
    int total_pixels = width * height;

    Complex *d_kSpace, *d_masked;
    int *d_row, *d_col; float *d_val;

    uploadImageDataToGPU(raw_img, total_pixels, &d_kSpace, &d_masked);
    initializeUndersamplingMask(total_pixels, &d_row, &d_col, &d_val);

    cout << "Processing: " << argv[1] << endl;
    executeReconstructionPipeline(width, height, total_pixels, d_kSpace, 
                                  d_masked, d_row, d_col, d_val);

    downloadAndSaveImage(d_masked, width, height);
    
    // Final Cleanup
    checkCuda(cudaFree(d_kSpace)); checkCuda(cudaFree(d_masked));
    checkCuda(cudaFree(d_row)); checkCuda(cudaFree(d_col)); 
    checkCuda(cudaFree(d_val));
    stbi_image_free(raw_img);

    cudaDeviceReset();
    return 0;
}
# Module 9 Assignment: MRI Reconstruction via Compressed Sensing

## What is this?

### Overview
This program addresses the "Scan Time" problem in MRI by implementing **Compressed Sensing** logic. Instead of requiring a full 20-minute scan, this tool simulates a high-speed scan by undersampling the frequency domain.

**Reference on background for this idea: https://pmc.ncbi.nlm.nih.gov/articles/PMC4984938/**

1.  **K-Space Transformation**: The program uses `cuFFT` to transform spatial image data into the frequency domain (k-space). Each pixel is treated as a sum of sine waves.
2.  **Sparse Masking**: It applies a Sparse Sampling Matrix (CSR format) using `cuSPARSE` to zero out 90% of the frequencies. This simulates the data loss of a high-speed scan.
3.  **Reconstruction**: The final organ image is reconstructed from this sparse frequency data using an inverse Fourier Transform.

NOTE: The issue isn't a compute time issue; it is an MRI machine hardware constraint. By keeping only 10-20% of the data, we reduce scan time enough for a patient to hold their breath or for a child to stay still.

### Use of Streams
The reconstruction pipeline utilizes a **custom CUDA Stream** to enable asynchronous execution:
1.  **Non-Default Stream**: By avoiding the NULL stream, the GPU can overlap the reconstruction compute with Host-to-Device data transfers.
2.  **Throughput**: In a clinical setting, this allows for the simultaneous processing of multiple "slices" of an organ, which is critical for real-time imaging feedback.


## How to run

### Prerequisites
To compile and run this program, you need:
* **CUDA Toolkit** (nvcc compiler and runtime libraries).
* **cuFFT** and **cuSPARSE** libraries (included with CUDA).
* `stb_image.h` and `stb_image_write.h` (single-header C libraries included in the repository).

### To run all tests:
This will automate the compilation and run the reconstruction pipeline against the provided sample images.
```bash
    ./run.sh
```

Or, you can manually run it via:
```bash
    wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    
    wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

    nvcc assignment.cu -o assignment -lcufft -lcusparse

    ./assignment brain.jpg

```

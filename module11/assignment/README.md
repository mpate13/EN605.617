# Module 11

## What is this?
This program is an OpenCL-based benchmark that performs a simple 2D coordinate transformation (scale + translate) on a large dataset. The goal is to demonstrate core and advanced OpenCL concepts while measuring performance on different workloads.

## Key Features
- Vectorized Processing
    - The kernel uses float2 to process (x, y) coordinate pairs in parallel, improving performance compared to using individual floats.
- Sub-Buffers
    - A single output buffer is split into two regions using a sub-buffer. This allows different parts of the data to be processed separately while still sharing the same memory.
- Two-Stage Execution
    - The workload is split in half:
        - First half writes directly to the main buffer
        - Second half writes to the sub-buffer using a global offset
    - This ensures the entire dataset is processed correctly.
- Profiling with Events
    - Kernel execution time is measured using OpenCL events, giving accurate timing directly from the device.
- Multi-Device Setup (Extensible)
    - The program creates command queues for all available devices (CPU/GPU), allowing for future expansion into multi-device execution. (didn't get it fully working this time around)


## How to Run
Note: did test on EC2 instance and Mac locally

### Run scaling tests
```bash
    ./run.sh
```

### Run individual run
```bash
    make clean
    make
    ./assignment <total_threads> <block_size>
```
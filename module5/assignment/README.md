# Module 5 Assignment
## What is this?
In hopes of expanding to a KMeans clustering problem (https://en.wikipedia.org/wiki/K-means_clustering), I have implemented the 'assignment' step in the single complex kernel method, using all 5 types of memory in a single problem.

1. Host memory is used to manage the initial dataset creation and storage in system RAM before any GPU processing begins. 
2. Global memory is utilized to provide a high-capacity storage area on the GPU for the large arrays of coordinate data and cluster results. 
3. Constant memory is implemented for the cluster centroids to take advantage of specialized hardware caching that speeds up simultaneous broadcast reads by all threads. 
4. Shared memory is used to stage point data locally within a block to reduce the high-latency overhead of repeated global memory accesses during distance calculations. 
5. Register memory is employed for local mathematical variables to ensure the core arithmetic operations happen at the maximum possible hardware frequency.

I also provided a CPU-only version for benchmarking purposes to see GPU performance

## How to run
### To run all tests
This will run tests for:
1. 64, 256, 1024 threads (1 block)
2. 1, 10, 100 blocks (1024 threads)
```bash
    ./run.sh
```
### To run an individual test:
```bash
    make
    ./assignment <blocks> <threads> 
```

## Where are things
1. Assignment code in `assignment.cu`
2. Proof of code running in `proof_of_run`
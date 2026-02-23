# #!/bin/bash
# 1. Clean and Build
make clean
make

echo "Starting Benchmarks..."
echo "Format: ./assignment <total_threads> <block_size>"

echo -e "\n[PHASE 1.1] 1 Block | 64 Threads"
./assignment 64 64

echo -e "\n[PHASE 1.2] 1 Block | 128 Threads"
./assignment 128 128

echo -e "\n[PHASE 1.3] 1 Block | 1024 Threads"
./assignment 1024 1024

# --- PHASE 2: Fixed Block Size, Increasing Workload ---
# Shows how the GPU scales out across the entire chip as data grows.
# Block size is fixed at 256 (a very common 'sweet spot' for occupancy).
echo -e "\n[PHASE 2.1] Fixed Block Size(256) | 10k Threads"
./assignment 10000 256

echo -e "\n[PHASE 2.2] Fixed Block Size (256) | 100k Threads"
./assignment 100000 256

echo -e "\n[PHASE 2.3] Fixed Block Size (256) | 1M Threads"
./assignment 1000000 256

# --- PHASE 3: Constant Work (1M Threads), Variable Block Size ---
# Demonstrates layout efficiency and Warp/Scheduler overhead.
echo -e "\n[PHASE 3.1] 1M Threads | Block Size 1024"
./assignment 1000000 1024

echo -e "\n[PHASE 3.2] 1M Threads | Block Size 128"
./assignment 1000000 128

echo -e "\n[PHASE 3.3] 1M Threads | Block Size 1"
./assignment 1000000 1
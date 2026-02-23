# #!/bin/bash
# 1. Clean and Build
make clean
make

echo "Starting Benchmarks..."
echo "Format: ./assignment <total_threads> <block_size>"

# --- PHASE 1: Single Block Scaling ---
echo -e "\n[TEST 1] Single Block: 64 total threads, Block Size 64"
./assignment 64 64

echo -e "\n[TEST 2] Single Block: 128 total threads, Block Size 128"
./assignment 128 128

echo -e "\n[TEST 3] Single Block: 1024 total threads, Block Size 1024"
./assignment 1024 1024

# --- PHASE 2: Constant Work (1M Threads), Variable Block Size ---
echo -e "\n[TEST 4] 1M Threads | Block Size 1024 (Efficient: 977 Blocks)"
./assignment 1000000 1024

echo -e "\n[TEST 5] 1M Threads | Block Size 128 (7,813 Blocks)"
./assignment 1000000 128

echo -e "\n[TEST 6] 1M Threads | Block Size 1 (1,000,000 Blocks)"
./assignment 1000000 1
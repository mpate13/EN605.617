# #!/bin/bash
# make clean
# make

# echo "Starting Benchmarks..."

# # Test 1: Baseline
# echo -e "\n[TEST 1] Testing baseline w/ one block (64 threads)."
# ./assignment 1 64

# # Test 2 & 3: Thread Scaling - increasing num threads within a single block
# echo -e "\n[TEST 2] Testing medium occupancy (256 threads)."
# ./assignment 1 256
# echo -e "\n[TEST 3] Testing maximum block occupancy (1024 threads)."
# ./assignment 1 1024

# # Test 4 & 5: Grid Scaling - multiple blocks
# echo -e "\n[TEST 4] Testing multiple blocks (10,240 total threads)."
# ./assignment 10 1024
# echo -e "\n[TEST 5] Testing large scale grid (102,400 total threads)."
# ./assignment 100 1024

#!/bin/bash
# Ensure the script stops if a command fails
set -e

# 1. Clean and Build
make clean
make

echo "Starting Benchmarks..."
echo "Format: ./assignment <Total_Threads> <Block_Size>"

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
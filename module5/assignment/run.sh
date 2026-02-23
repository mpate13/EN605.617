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
echo "Rebuilding project..."
make clean
make

echo "Starting Benchmarks..."
echo "Format: ./assignment <Total_Threads> <Block_Size>"

# TEST 1: Baseline (Requirement: Minimum 64 threads)
echo -e "\n[TEST 1] Baseline: 64 total threads, Block Size 64 (1 Block)"
./assignment 64 64

# TEST 2 & 3: Varying Block Size (Requirement: 2 additional block sizes)
echo -e "\n[TEST 2] Medium Block Size: 1024 total threads, Block Size 256 (4 Blocks)"
./assignment 1024 256

echo -e "\n[TEST 3] Small Block Size: 1024 total threads, Block Size 32 (32 Blocks)"
./assignment 1024 32

# TEST 4 & 5: Varying Total Threads (Requirement: 2 additional thread counts)
echo -e "\n[TEST 4] Large Workload: 10,000 total threads, Block Size 512 (~20 Blocks)"
./assignment 10000 512

echo -e "\n[TEST 5] Stress Test: 1,000,000 total threads, Block Size 1024 (~977 Blocks)"
./assignment 1000000 1024

echo -e "\nBenchmarks Complete. Please screen capture this output for the bonus."
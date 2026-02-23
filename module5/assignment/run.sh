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

# --- CATEGORY 1: Scaling Threads within a SINGLE BLOCK ---
# This proves the code works at different occupancy levels on 1 SM.
echo -e "\n[TEST 1] 1 Block | 64 Threads (Baseline)"
./assignment 64 64

echo -e "\n[TEST 2] 1 Block | 256 Threads"
./assignment 256 256

echo -e "\n[TEST 3] 1 Block | 1024 Threads (Max Block Size)"
./assignment 1024 1024

# --- CATEGORY 2: Constant Work, Variable Block Size ---
# This proves the code handles different Grid/Block layouts for the same total work.
# Total threads is kept constant at 4096.
echo -e "\n[TEST 4] 4096 Total Threads | Block Size 64  (64 Blocks)"
./assignment 4096 64

echo -e "\n[TEST 5] 4096 Total Threads | Block Size 256 (16 Blocks)"
./assignment 4096 256

echo -e "\n[TEST 6] 4096 Total Threads | Block Size 1024 (4 Blocks)"
./assignment 4096 1024
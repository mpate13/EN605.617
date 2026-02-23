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

# --- CATEGORY 1: Small Scale (GPU Overhead dominates) ---
echo -e "\n[TEST 1] Small Scale: 1,024 Threads | Block Size 256"
./assignment 1024 256

# --- CATEGORY 2: Medium Scale (GPU begins to pull away) ---
echo -e "\n[TEST 2] Medium Scale: 102,400 Threads | Block Size 256"
./assignment 102400 256

# --- CATEGORY 3: Large Scale (GPU Power Showcase) ---
echo -e "\n[TEST 3] Large Scale: 1,048,576 Threads | Block Size 256"
./assignment 1048576 256

# --- CATEGORY 4: Varying Block Size at Large Scale ---
# Keeping work the same (1M points) but changing hardware layout
echo -e "\n[TEST 4] 1M Threads | Block Size 64 (High Block Count)"
./assignment 1048576 64

echo -e "\n[TEST 5] 1M Threads | Block Size 1024 (Low Block Count)"
./assignment 1048576 1024
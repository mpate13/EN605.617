#!/bin/bash

# 1. Compile the code
# -O3 for optimization, -lcudart for the runtime library
echo "--- Compiling CUDA Program ---"
nvcc assignment.cu -o assignment

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful. Starting Test Harness..."
echo "=================================================================="

# Define Constants
FIXED_THREADS=1048576
FIXED_BLOCK_SIZE=256

# Test Set 1: Constant Threads, Different Block Sizes
echo "TEST SET A: Fixed Threads ($FIXED_THREADS), Varying Block Sizes"
for BLOCK_SIZE in 128 256 512
do
    ./assignment $FIXED_THREADS $BLOCK_SIZE
done

echo ""
# Test Set 2: Different Total Threads, Fixed Block Size
echo "TEST SET B: Varying Total Threads, Fixed Block Size ($FIXED_BLOCK_SIZE)"
for TOTAL_THREADS in 524288 2097152 4194304
do
    ./encrypt_test $TOTAL_THREADS $FIXED_BLOCK_SIZE
done

echo "=================================================================="
echo "Testing Complete."
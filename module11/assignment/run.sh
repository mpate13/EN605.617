#!/bin/bash

# Compile the latest code
make clean
make

echo "------------------------------------------------"
echo "Starting OpenCL Benchmark Tests"
echo "------------------------------------------------"

# PART 1: VARYING GLOBAL SIZE (Total Threads) ---
# Keeping Local Size (Block Size) constant at 64
echo "PART 1: Scaling Total Workload (Block Size = 64)"
echo "------------------------------------------------"

for G in 1024 1048576 4194304
do
    echo "Testing Global Size: $G | Local Size: 64"
    ./assignment $G 64
    echo ""
done

# PART 2: VARYING LOCAL SIZE (Block Size) ---
# Keeping Global Size (Total Threads) constant at 1048576
echo "PART 2: Scaling Block Size (Total Threads = 1048576)"
echo "------------------------------------------------"

for L in 1 64 256
do
    echo "Testing Global Size: 1048576 | Local Size: $L"
    ./assignment 1048576 $L
    echo ""
done

echo "------------------------------------------------"
echo "Tests Complete."
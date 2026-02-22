#!/bin/bash
make clean
make

echo "Starting CUDA Memory Assignment Benchmarks..."

# Test 1: Baseline
echo -e "\n[TEST 1] Testing baseline w/ one block (64 threads)."
./assignment 1 64

# Test 2 & 3: Thread Scaling - increasing num threads within a single block
echo -e "\n[TEST 2] Testing medium occupancy (256 threads)."
./assignment 1 256
echo -e "\n[TEST 3] Testing maximum block occupancy (1024 threads)."
./assignment 1 1024

# Test 4 & 5: Grid Scaling - multiple blocks
echo -e "\n[TEST 4] Testing multiple blocks (10,240 total threads)."
./assignment 10 1024
echo -e "\n[TEST 5] Testing large scale grid (102,400 total threads)."
./assignment 100 1024
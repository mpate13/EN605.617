#!/bin/bash

# Compile the latest code
make clean
make

echo "------------------------------------------------"
echo "Starting OpenCL Benchmark Tests"
echo "------------------------------------------------"

# Test 1: Standard Small Run
echo "Test 1: Small Array (1024), Balanced Local Size (64)"
./assignment 1024 64
echo ""

# Test 2: Large Alignment Run (Power of 2)
echo "Test 2: Large Array (1048576), Balanced Local Size (64)"
./assignment 1048576 64
echo ""

# Test 3: Large Alignment Run (Small Local Size)
echo "Test 3: Large Array (1048576), Worst-Case Local Size (1)"
./assignment 1048576 1
echo ""

echo "------------------------------------------------"
echo "Tests Complete."
#!/bin/bash

# Compile first
make clean
make

# Check if compile worked
if [ ! -f assignment ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "--- Starting Block Size Tests ---"
./assignment 1048576 128
echo "---------------------------------------------------------------------"
./assignment 1048576 512
echo "---------------------------------------------------------------------"
./assignment 1048576 1024
echo "---------------------------------------------------------------------"

echo "--- Starting Thread Count Tests ---"
./assignment 524288 256
echo "---------------------------------------------------------------------"
./assignment 2097152 256
echo "---------------------------------------------------------------------"
./assignment 4194304 256
echo "---------------------------------------------------------------------"
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
./assignment 1048576 256
./assignment 1048576 512

echo "--- Starting Thread Count Tests ---"
./assignment 524288 256
./assignment 2097152 256
./assignment 4194304 256
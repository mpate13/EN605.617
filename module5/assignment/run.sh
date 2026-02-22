#!/bin/bash
make clean
make

echo "Executing rubric-required runs..."
# Minimum 64 threads test
./assignment 1 64

# Additional thread sizes
./assignment 1 256
./assignment 1 1024

# Additional block sizes
./assignment 10 1024
./assignment 100 1024
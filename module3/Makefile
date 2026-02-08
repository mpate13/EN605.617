# CUDA compiler
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# C compiler
CC = gcc

# --- Targets ---
all: assignment assignment_cpu

# GPU version (from assignment.cu)
assignment: assignment.cu
	$(NVCC) -o assignment assignment.cu

# CPU version (from assignment.c)
assignment_cpu: assignment.c
	$(CC) -o assignment_cpu assignment.c

# Clean both executables
clean:
	rm -f assignment assignment_cpu

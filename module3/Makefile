# CUDA compiler
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# C compiler
CC = gcc

# --- Targets ---
all: assignment.exe assignment_cpu.exe

# GPU version (from assignment.cu)
assignment.exe: assignment.cu
	$(NVCC) -o assignment.exe assignment.cu

# CPU version (from assignment.c)
assignment_cpu.exe: assignment.c
	$(CC) -o assignment_cpu.exe assignment.c

# Clean both executables
clean:
	rm -f assignment.exe assignment_cpu.exe

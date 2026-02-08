# CUDA compiler
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_LIB = -L$(CUDA_PATH)/lib -lcudart

# C compiler
CC = gcc

# make all
all: assignment.exe assignment_cpu.exe

# GPU version
assignment.exe: assignment.cu
	$(NVCC) assignment.cu $(CUDA_LIB) -o assignment.exe

# CPU version
assignment_cpu.exe: assignment.c
	$(CC) -o assignment_cpu.exe assignment.c

# Clean both executables
clean:
	rm -f assignment.exe assignment_cpu.exe

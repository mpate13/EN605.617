CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# Targets
all: assignment.exe assignment_cpu.exe

# GPU version (from assignment.cu)
assignment.exe: assignment.cu
	$(NVCC) -O2 -o $@ $^

# CPU version (from assignment.c)
assignment_cpu.exe: assignment.c
	gcc -O2 -o $@ $^

# Clean both executables
clean:
	rm -f assignment.exe assignment_cpu.exe

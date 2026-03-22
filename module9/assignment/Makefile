# Compiler and Flags
NVCC = nvcc
# -diag-suppress 550 silences the 'variable set but not used' warnings in stb
NVCC_FLAGS = -diag-suppress 550
LIBS = -lcufft -lcusparse

# Target executable name
TARGET = assignment

# Source files
SRC = assignment.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) reconstructed_mri.jpg
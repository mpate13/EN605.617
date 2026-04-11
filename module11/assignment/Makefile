# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -Wall -O3
# Frameworks for macOS
LDFLAGS = -framework OpenCL

TARGET = assignment
SRC = assignment.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
CXX = g++
CXXFLAGS = -Wall -O3
TARGET = assignment
SRC = assignment.cpp

# Check if we are on Apple or Linux
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    # macOS settings
    LDFLAGS = -framework OpenCL
else
    # Linux/AWS settings
    LDFLAGS = -lOpenCL
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
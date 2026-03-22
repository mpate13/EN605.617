#!/bin/bash

# 1. Download dependencies if they don't exist
if [ ! -f "stb_image.h" ]; then
    echo "Downloading stb_image.h..."
    wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
fi

if [ ! -f "stb_image_write.h" ]; then
    echo "Downloading stb_image_write.h..."
    wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
fi

# 2. Compile the project
echo "Compiling with nvcc..."
make clean
make

# 3. Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    
    # 4. Run the program (Assuming you have an image named 'brain.jpg')
    if [ -f "brain.jpg" ]; then
        echo "Running reconstruction on brain.jpg..."
        ./assignment brain.jpg
    else
        echo "Note: Place an 'brain.jpg' in this folder to run the test."
    fi
else
    echo "Compilation failed."
    exit 1
fi
sudo apt install python3-pip
# This installs the cuDNN 9 library specifically for CUDA 12
sudo apt-get install -y libcudnn9-dev-cuda-12
pip install pandas numpy scipy
python3 preprocess.py


nvcc assignment.cu -o recommender \
-I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu \
-lcudnn -lcusparse



wget -O brain.jpg https://raw.githubusercontent.com/opencv/opencv/master/samples/data/gradient.png
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h


This is the very latest version of the README file for the basic Introduction To GPU Programming Course.


Some directions for spinning up aws
1. go to aws.com and sign into console https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#Instances:
2. vpc
3. launch ec2 instance
    - name: jhu_gpu
    - ubuntu default
    - g4dn.xlarge
    - use jhu_gpu keys (in downloads)
    - use my ip
    - storage size: 40 gib, default type is fine
    - launch instance
4. click on instance id
5. copy public dns (ipv4)
6. go to directory with ssh keys (~/Downloads)
7. ssh -i jhu_gpu.pem ubuntu@{public dns address}
8. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
9. sudo dpkg -i cuda-keyring_1.1-1_all.deb
10. sudo apt-get update
11. sudo apt-get upgrade -y
12. sudo apt-get install -y cuda opencl-headers build-essential cmake nvidia-cuda-toolkit libboost-all-dev
13. git clone https://github.com/mpate13/EN605.617.git

some commands:
- nvidia-smi
- nvcc to compile


to shutdown:
1. sudo shutdown -h now
2. go back to aws browswer
3. stop AND delete instance to stop spending money
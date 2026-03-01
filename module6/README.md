# Module 6 Assignment
## What is this?

### Overview
The program performs two main tasks:
1. Key Generation: It uses a mathematical "hashing" algorithm to create unique encryption keys for every piece of data.
2. Data Encryption: It uses those keys to scramble the data into a secure format using bitwise rotations and XOR operations.

NOTE: There is also a CPU implementation included for timing tests

### Use of Streams
1. Stream A (Key Generation): This lane is dedicated to calculating the unique encryption keys.
2. Stream B (Data Handling): This lane works at the same time as Stream A to move data from memory onto the GPU. It is also responsible for the final encryption and moving the finished data back.

### Use of Events
1. Start/Stop Events: Used for timing
2. Ready Event: This is a "traffic signal." It is placed at the end of Stream A. Stream B is told to watch for this signal and cannot start the encryption until the signal says the keys are 100% finished.


## How to run
### To run all tests
This will run tests for:
1. 128, 512, 1024 block size (1048576 threads)
2. 524288, 2097152, 4194304 threads (256 block size)
```bash
    ./run.sh
```
### To run an individual test:
```bash
    make
    ./assignment <total_threads> <block_size> 
```

## Where are things
1. Assignment code in `assignment.cu`
2. Proof of code running in `proof_of_run`
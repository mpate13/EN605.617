#include <stdio.h>
#include <stdlib.h>

#define KERNEL_LOOP 1048576
#define KERNEL_SIZE 128

__host__ void wait_exit(void)
{
        char ch;

        printf("\nPress any key to exit");
        ch = getchar();
}

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
        for(unsigned int i=0; i < KERNEL_LOOP; i++)
        {
                host_data_ptr[i] = (unsigned int) rand();
        }
}

// __global__ void test_gpu_register(unsigned int * const data, const unsigned int num_elements)
// {
//         const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
//         if(tid < num_elements)
//         {
//                 unsigned int d_tmp = data[tid];
//                 d_tmp = d_tmp * 2;
//                 data[tid] = d_tmp;
//         }
// }

__global__ void test_gpu_register(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements)
    {
        unsigned int d0 = data[tid];
        
        // Force the GPU to hold many values at once
        unsigned int d1 = d0 * 2;
        unsigned int d2 = d1 + d0;
        unsigned int d3 = d2 * d1;
        unsigned int d4 = d3 + d2;
        unsigned int d5 = d4 * d3;
        unsigned int d6 = d5 + d4;
        unsigned int d7 = d6 * d5;
        unsigned int d8 = d7 + d6;
        unsigned int d9 = d8 * d7;
        unsigned int d10 = d9 + d8;

        // Final calculation using the chain
        data[tid] = d10;
    }
}

__host__ void gpu_kernel(void)
{
        const unsigned int num_elements = KERNEL_LOOP;
        const unsigned int num_threads = KERNEL_SIZE;
        const unsigned int num_blocks = (num_elements + num_threads - 1)/num_threads;
        const unsigned int num_bytes = num_elements * sizeof(unsigned int);

        unsigned int * data_gpu;

        unsigned int host_packed_array[num_elements];
        unsigned int host_packed_array_output[num_elements];

        cudaMalloc(&data_gpu, num_bytes);

        generate_rand_data(host_packed_array);

        cudaMemcpy(data_gpu, host_packed_array, num_bytes,cudaMemcpyHostToDevice);

        // test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements);
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start); // Start timer

        test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements);

        cudaEventRecord(stop); // Stop timer
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("\nRegister Kernel Time: %f ms\n", milliseconds);



        cudaDeviceSynchronize();        // Wait for the GPU launched work to complete
        cudaGetLastError();

        cudaMemcpy(host_packed_array_output, data_gpu, num_bytes,cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_elements; i++){
                printf("Input value: %x, device output: %x\n",host_packed_array[i], host_packed_array_output[i]);
        }

        cudaFree((void* ) data_gpu);
        cudaDeviceReset();
//        wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	gpu_kernel();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return EXIT_SUCCESS;
}


/*
default: kernel loop 2048
Register Kernel Time: 12.710368 ms


kernel loop 1m


*/
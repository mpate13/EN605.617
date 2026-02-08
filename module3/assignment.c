#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Branch-free CPU transformation function.
 * 
 * For each element in the input array `x`, compute y[i] = x[i] * 2 + 3.
 *
 * @param x Input array
 * @param y Output array
 * @param N Number of elements to process
 */
void transform_cpu(int *x, int *y, int N)
{
	// Iterate through each item and do the computation
    for(int i=0; i<N; i++)
        y[i] = x[i] * 2 + 3;
}


/**
 * @brief Branched CPU transformation function.
 * 
 * For each element in the input array `x`, check if it is even or odd:
 * - Even: y[i] = x[i] * 2
 * - Odd:  y[i] = x[i] * 3
 *
 * @param x Input array
 * @param y Output array
 * @param N Number of elements to process
 */
void transform_branch_cpu(int *x, int *y, int N)
{
	// Iterate through each item
    for(int i=0; i<N; i++)
    {
		// Same branch logic as GPU version
        if(x[i] % 2 == 0)
            y[i] = x[i] * 2;
        else
            y[i] = x[i] * 3;
    }
}

/**
 * @brief Main function to execute CPU transformations and measure execution times.
 * 
 * Accepts one optional command-line argument to set the total number of elements:
 * ./assignment_cpu.exe [totalThreads]
 * 
 * Executes both branch-free and branched versions and prints timings.
 */
int main(int argc, char** argv)
{
    // Default total threads - help with testing
    int totalThreads = 1 << 20;  // 1,048,576 elements

	// Get the number of threads if there is a user input value
    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }

    printf("CPU: Total elements = %d\n", totalThreads);

    // Allocate and initialize arrays
    int *x = (int*)malloc(totalThreads * sizeof(int));
    int *y = (int*)malloc(totalThreads * sizeof(int));
    int *y_branch = (int*)malloc(totalThreads * sizeof(int));

	// Initialize the array
    for(int i=0; i<totalThreads; i++)
        x[i] = rand() % 100;

    // Branch free test
    clock_t start = clock();
    transform_cpu(x, y, totalThreads);
    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU branch-free time: %f ms\n", cpu_time);

    // Branched test
    start = clock();
    transform_branch_cpu(x, y_branch, totalThreads);
    end = clock();
    double cpu_branch_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU branchy time: %f ms\n", cpu_branch_time);

    // Clean up
    free(x);
    free(y);
    free(y_branch);

    return 0;
}

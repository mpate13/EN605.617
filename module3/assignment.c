// assignment.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Branch-free CPU transformation
void transform_cpu(int *x, int *y, int N)
{
    for(int i=0; i<N; i++)
        y[i] = x[i] * 2 + 3;
}

// Branchy CPU transformation
void transform_branch_cpu(int *x, int *y, int N)
{
    for(int i=0; i<N; i++)
    {
        if(x[i] % 2 == 0)
            y[i] = x[i] * 2;
        else
            y[i] = x[i] * 3;
    }
}

int main(int argc, char** argv)
{
    // Default total threads
    int totalThreads = 1 << 20;  // 1,048,576

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }

    printf("CPU: Total elements = %d\n", totalThreads);

    // Allocate and initialize arrays
    int *x = (int*)malloc(totalThreads * sizeof(int));
    int *y = (int*)malloc(totalThreads * sizeof(int));
    int *y_branch = (int*)malloc(totalThreads * sizeof(int));

    for(int i=0; i<totalThreads; i++)
        x[i] = rand() % 100;

    // --- CPU branch-free ---
    clock_t start = clock();
    transform_cpu(x, y, totalThreads);
    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU branch-free time: %f ms\n", cpu_time);

    // --- CPU branchy ---
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

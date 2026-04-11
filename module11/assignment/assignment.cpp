#define CL_TARGET_OPENCL_VERSION 220
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/**
 * CONCEPTS USED IN THIS ASSIGNMENT
 * - Context & Multiple Queues: Manages the global environment and creates 
 * command lanes for every available device (GPU/CPU).
 * - Buffers & Sub-Buffers: Allocates a parent memory block (m[1]) and 
 * carves out a 'cl_buffer_region' (m[2]). This ensures the kernel 
 * writes only to a specific section in memory.
 * - Vectors (float2): Uses native GPU vector types to process (x, y) 
 * pairs simultaneously, doubling mathematical throughput.
 * - MapBuffer: Maps GPU memory directly to a host pointer. 
 * Eliminates slow transfers, optimized for unified memory.
 */

const int ARRAY_SIZE = 1024;

cl_context CreateContext() {
    cl_uint numPlatforms;
    cl_platform_id platformId;
    clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (numPlatforms <= 0) return NULL;

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0
    };
    return clCreateContextFromType(props, CL_DEVICE_TYPE_ALL, NULL, NULL, NULL);
}

///
//  Create command queues for ALL devices found in the context
//
std::vector<cl_command_queue> CreateQueues(cl_context ctx, 
                                          std::vector<cl_device_id> &devs) {
    size_t size;
    clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &size);
    int num = size / sizeof(cl_device_id);
    
    devs.resize(num);
    clGetContextInfo(ctx, CL_CONTEXT_DEVICES, size, devs.data(), NULL);

    std::vector<cl_command_queue> queues;
    for (int i = 0; i < num; i++) {
        // Added CL_QUEUE_PROFILING_ENABLE to allow timing
        cl_command_queue q = clCreateCommandQueue(ctx, devs[i], CL_QUEUE_PROFILING_ENABLE, NULL);
        if (q) queues.push_back(q);
    }
    return queues;
}

cl_program CreateProgram(cl_context context, 
    cl_device_id device, const char* fileName) {
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects: [0]=Input, [1]=Output Parent, [2]=Sub-Buffer
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], size_t g_size) {
    cl_int err;
    size_t full_bytes = g_size * 2 * sizeof(float);
    size_t half_bytes = g_size * sizeof(float); // Must be alignment multiple

    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                   full_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return false;
    }

    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                   full_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    
    cl_buffer_region region = { half_bytes, half_bytes };
    memObjects[2] = clCreateSubBuffer(memObjects[1], CL_MEM_WRITE_ONLY,
                                      CL_BUFFER_CREATE_TYPE_REGION, 
                                      &region, &err);

    return (err == CL_SUCCESS);
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, std::vector<cl_command_queue> commandQueues,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    
    for (size_t j = 0; j < commandQueues.size(); j++) {
        if (commandQueues[j] != 0)
            clReleaseCommandQueue(commandQueues[j]);
    }

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}

int main(int argc, char** argv) {
    cl_int err;
    cl_mem m[3] = {0, 0, 0};
    // global size = total threads, l_size = block size
    size_t g_size = (argc > 1) ? std::stoul(argv[1]) : ARRAY_SIZE;
    size_t l_size = (argc > 2) ? std::stoul(argv[2]) : 64;

    cl_context ctx = CreateContext();
    std::vector<cl_device_id> devs;
    std::vector<cl_command_queue> qs = CreateQueues(ctx, devs);
    
    cl_program prog = CreateProgram(ctx, devs[0], "assignment.cl");
    cl_kernel kernel = clCreateKernel(prog, "transform_kernel", NULL);

    std::vector<float> h_in(g_size * 2, 1.0f);
    size_t bytes = g_size * 2 * sizeof(float);

    if (!CreateMemObjects(ctx, m, g_size)) {
        Cleanup(ctx, qs, prog, kernel, m);
        return 1;
    }

    clEnqueueWriteBuffer(qs[0], m[0], CL_TRUE, 0, bytes, h_in.data(), 0,0,0);

    float scale = 5.0f; 
    cl_float2 trans = {{20.0f, 20.0f}};
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &m[0]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &m[2]);
    clSetKernelArg(kernel, 2, sizeof(float), &scale);
    clSetKernelArg(kernel, 3, sizeof(cl_float2), &trans);

    size_t work_size = g_size / 2;
    
    // Create an event to track kernel execution time
    cl_event profileEvent;
    clEnqueueNDRangeKernel(qs[0], kernel, 1, NULL, &work_size, &l_size, 0, 0, &profileEvent);

    // Wait for kernel to finish before calculating time
    clWaitForEvents(1, &profileEvent);

    cl_ulong start, end;
    clGetEventProfilingInfo(profileEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(profileEvent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double nanoSeconds = end - start;

    float* ptr = (float*)clEnqueueMapBuffer(qs[0], m[1], CL_TRUE, CL_MAP_READ, 
                                            0, bytes, 0, 0, 0, &err);
    
    std::cout << "Index 0: " << ptr[0] << " | Index " << g_size << ": " << ptr[g_size] << "\n";
    std::cout << "Kernel Execution Time: " << nanoSeconds / 1000000.0 << " ms\n";
    
    clReleaseEvent(profileEvent);
    clEnqueueUnmapMemObject(qs[0], m[1], ptr, 0, NULL, NULL);
    Cleanup(ctx, qs, prog, kernel, m);
    return 0;
}
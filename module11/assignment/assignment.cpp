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
    cl_int err;
    err = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0) return NULL;

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0
    };
    return clCreateContextFromType(props, CL_DEVICE_TYPE_ALL, NULL, NULL, &err);
}

///
//  Create command queues for ALL devices found in the context
// note: had to use a different clCreateCommandQueue to 
// eliminate deprication warnings
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
        cl_command_queue q = NULL;
        cl_int err;
        // Check for macOS or legacy OpenCL 1.2
#if defined(__APPLE__) || defined(__MACOSX)
        // Use the legacy 1.2 function
        q = clCreateCommandQueue(ctx, devs[i], CL_QUEUE_PROFILING_ENABLE, &err);
#else
        // Use the modern 2.0+ function for Linux/Windows
        cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 
            CL_QUEUE_PROFILING_ENABLE, 0 };
        q = clCreateCommandQueueWithProperties(ctx, devs[i], props, &err);
#endif
        
        if (err == CL_SUCCESS) queues.push_back(q);
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
                                        NULL, &errNum);
    if (errNum != CL_SUCCESS) {
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
bool CreateMemObjects(cl_context context, cl_device_id device, 
    cl_mem memObjects[3], size_t g_size) {
    cl_int err;
    cl_uint alignment;
    
    clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, 
        sizeof(cl_uint), &alignment, NULL);
    
    size_t alignBytes = alignment / 8; 

    size_t full_bytes = g_size * 2 * sizeof(float);
    size_t requested_half = g_size * sizeof(float);

    size_t aligned_offset = (requested_half / alignBytes) * alignBytes;

    memObjects[0] = 
        clCreateBuffer(context, CL_MEM_READ_ONLY, full_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    memObjects[1] = 
        clCreateBuffer(context, CL_MEM_READ_WRITE, full_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    
    cl_buffer_region region = { aligned_offset, full_bytes - aligned_offset };
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

/**
 * Sets the static kernel arguments (Input buffer, scale, and translation).
 * Returns CL_SUCCESS if all arguments were set correctly.
 */
cl_int SetStaticKernelArgs(cl_kernel kernel, cl_mem inputMem, 
                            float scale, cl_float2 translation) {
    cl_int err;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputMem);
    err |= clSetKernelArg(kernel, 2, sizeof(float), &scale);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_float2), &translation);
    return err;
}

/**
 * Dispatches the workload in two stages:
 * 1. Lower half directly into the parent output buffer.
 * 2. Upper half into the sub-buffer using a global offset.
 */
cl_int ExecuteTwoStageTransform(cl_command_queue queue, cl_kernel kernel, 
                                cl_mem parentOut, cl_mem subBufferOut,
                                size_t total_size, size_t local_size, 
                                cl_event* profileEvent) {
    cl_int err;
    size_t half_work = total_size / 2;
    size_t offset = half_work;

    // Lower Half
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &parentOut);
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
        &half_work, &local_size, 0, 0, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }

    // Upper Half (Using Sub-Buffer and Offset)
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &subBufferOut);
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &half_work, 
        &local_size, 0, 0, profileEvent);
    
    return err;
}

/**
 * Maps the buffer to host memory, prints verification indices and timing, 
 * then unmaps the buffer.
 */
void PrintResultsAndCleanup(cl_command_queue queue, cl_mem outputMem, 
                            cl_event profileEvent, size_t total_size) {
    cl_int err;
    size_t bytes = total_size * 2 * sizeof(float);
    cl_ulong start, end;

    clWaitForEvents(1, &profileEvent);
    clGetEventProfilingInfo(profileEvent, CL_PROFILING_COMMAND_START, 
        sizeof(start), &start, NULL);
    clGetEventProfilingInfo(profileEvent, CL_PROFILING_COMMAND_END, 
        sizeof(end), &end, NULL);

    float* ptr = (float*)clEnqueueMapBuffer(queue, outputMem, CL_TRUE, 
        CL_MAP_READ, 0, bytes, 0, 0, 0, &err);
    if (err == CL_SUCCESS) {
        std::cout << "Index 0: " << ptr[0] << " | Index " << 
            total_size << ": " << ptr[total_size] << "\n";
        std::cout << "Kernel Execution Time (GPU): " << 
            (end - start) / 1000000.0 << " ms\n";
        clEnqueueUnmapMemObject(queue, outputMem, ptr, 0, NULL, NULL);
    }
    clReleaseEvent(profileEvent);
}

int main(int argc, char** argv) {
    cl_mem m[3] = {0, 0, 0};
    size_t g_size = (argc > 1) ? std::stoul(argv[1]) : ARRAY_SIZE;
    size_t l_size = (argc > 2) ? std::stoul(argv[2]) : 64;

    cl_context ctx = CreateContext();
    std::vector<cl_device_id> devs;
    std::vector<cl_command_queue> qs = 
        (ctx) ? CreateQueues(ctx, devs) : std::vector<cl_command_queue>();
    if (qs.empty()) return 1;

    cl_program prog = CreateProgram(ctx, devs[0], "assignment.cl");
    cl_kernel kernel = 
        (prog) ? clCreateKernel(prog, "transform_kernel", NULL) : NULL;
    if (!kernel || !CreateMemObjects(ctx, devs[0], m, g_size)) {
        Cleanup(ctx, qs, prog, kernel, m);
        return 1;
    }

    // Prepare data and static arguments
    std::vector<float> h_in(g_size * 2, 1.0f);
    clEnqueueWriteBuffer(qs[0], m[0], CL_TRUE, 0, g_size * 2 * sizeof(float), 
        h_in.data(), 0, 0, 0);
    
    if (SetStaticKernelArgs(kernel, m[0], 5.0f, 
        {{20.0f, 20.0f}}) != CL_SUCCESS) {
        return 1;
    }

    // Execute and Profile
    cl_event profileEvent;
    if (ExecuteTwoStageTransform(qs[0], kernel, m[1], m[2], 
        g_size, l_size, &profileEvent) != CL_SUCCESS) {
            return 1;
    }

    PrintResultsAndCleanup(qs[0], m[1], profileEvent, g_size);

    Cleanup(ctx, qs, prog, kernel, m);
    return 0;
}
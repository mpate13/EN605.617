// assignment.cl
// Performs a 2D Scale and Translation: Output = (Input * Scale) + Trans
// Demonstrates vectorized math on float2 (x, y) pairs.
__kernel void transform_kernel(__global const float2* input,
                               __global float2* output,
                               const float scale,
                               const float2 translation)
{
    // Absolute ID (0 for first call, 512 for second call)
    int id = get_global_id(0);
    
    // Relative ID (Always 0-511 relative to the buffer provided)
    int out_idx = id - get_global_offset(0);
    
    float2 position = input[id];
    output[out_idx] = (position * scale) + translation;
}
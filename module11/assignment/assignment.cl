// assignment.cl
// Performs a 2D Scale and Translation: Output = (Input * Scale) + Trans
// Demonstrates vectorized math on float2 (x, y) pairs.
__kernel void transform_kernel(__global const float2* input,
                               __global float2* output,
                               const float scale,
                               const float2 translation)
{
    // Get the unique ID for this work-item
    int id = get_global_id(0);
    
    // Process both X and Y in a single vectorized operation
    float2 position = input[id];
    output[id] = (position * scale) + translation;
}
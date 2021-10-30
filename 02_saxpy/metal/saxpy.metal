#include <metal_stdlib>

using namespace metal;

struct saxpy_constants
{
    uint  num_elements;
};

kernel void saxpy(

    device const float*           X [[ buffer(0) ]],
    device       float*           Y [[ buffer(1) ]],
    device       float&           a [[ buffer(2) ]],
    device       saxpy_constants& c [[ buffer(3) ]],

    const        uint             thread_position_in_grid [[ thread_position_in_grid ]],
    const        uint             threads_per_grid        [[ threads_per_grid ]]
) {
    for ( uint i = thread_position_in_grid; i < c.num_elements; i += threads_per_grid ) {
    
        Y[i] = X[i] * a + Y[i];
    }
}

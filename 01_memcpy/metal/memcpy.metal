#include <metal_stdlib>

using namespace metal;

struct memcpy_constants
{
    uint  num_elements;
};


kernel void my_memcpy(

    device const int*            in                             [[ buffer(0) ]],

    device       int*            out                            [[ buffer(1) ]],

    device const memcpy_constants& c                            [[ buffer(2) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]]
) {
    if ( thread_position_in_grid < c.num_elements ) {

        out[ thread_position_in_grid ] = in[ thread_position_in_grid ];
    }
}

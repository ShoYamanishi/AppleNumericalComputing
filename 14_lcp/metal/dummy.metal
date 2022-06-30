#include <metal_stdlib>

kernel void dummy (
    const uint thread_position_in_grid [[ thread_position_in_grid ]]
) {
    return;
}


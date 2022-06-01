#include <metal_stdlib>

using namespace metal;


typedef struct _bitonic_sort_constants
{
    uint  total_num_elements; // must be 2^x. threadgroups_per_grid must be total_num_elements / 2
    uint  swap_span;
    uint  block_size; // A unit of consecutive sub-array of size 2^x that are sorted in the same direction. 
                      // The maximum/initial span for swapping is (2-1)^x.
} bitonic_sort_constants;


kernel void bitonic_sort_one_swap_span_int(
    device       int*                     target_array             [[ buffer( 0) ]],
    device const bitonic_sort_constants&  constants                [[ buffer( 1) ]],
    const        uint                     thread_position_in_grid  [[ thread_position_in_grid ]],
    const        uint                     threadgroups_per_grid    [[ threadgroups_per_grid ]]
) {
    if ( thread_position_in_grid < (constants.total_num_elements / 2) ) {

        const int  half_block_size          = constants.block_size / 2;
        const int  block_index              = thread_position_in_grid / half_block_size;
        const int  thread_index_in_block    = thread_position_in_grid % half_block_size;

        const bool is_upward                = block_index % 2 == 0; // 0, 2, 4, 6,...

        const int  subblock_size            = constants.swap_span * 2;
        //const int  num_swaps_per_subblock   = constants.block_size / subblock_size;
        const int  subblock_index           = thread_index_in_block / (subblock_size / 2);
        const int  thread_index_in_subblock = thread_index_in_block % (subblock_size / 2);

        const int  lower_array_index =   block_index * constants.block_size
                                       + subblock_index * subblock_size
                                       + thread_index_in_subblock;

        const int  upper_array_index = lower_array_index + constants.swap_span;

        if(    ( ( is_upward) && ( target_array[lower_array_index] > target_array[upper_array_index] ) )
            || ( (!is_upward) && ( target_array[lower_array_index] < target_array[upper_array_index] ) ) ) {

            const int saved                 = target_array[lower_array_index];
            target_array[lower_array_index] = target_array[upper_array_index];
            target_array[upper_array_index] = saved;
        }
    }
}


kernel void bitonic_sort_multiple_swap_spans_down_to_2_int(

    device       int*                     target_array             [[ buffer( 0) ]],
    device const bitonic_sort_constants&  constants                [[ buffer( 1) ]],
    const        uint                     thread_position_in_grid  [[ thread_position_in_grid ]],
    const        uint                     threadgroups_per_grid    [[ threadgroups_per_grid ]]
) {

    const int  half_block_size          = constants.block_size / 2;
    const int  block_index              = thread_position_in_grid / half_block_size;
    const int  thread_index_in_block    = thread_position_in_grid % half_block_size;

    const bool is_upward                = block_index % 2 == 0; // 0, 2, 4, 6,...

    for ( int swap_span = constants.swap_span; swap_span >= 1; swap_span /= 2 ) {

        const int  subblock_size            = swap_span * 2;
        //const int  num_swaps_per_subblock   = constants.block_size / subblock_size;
        const int  subblock_index           = thread_index_in_block / (subblock_size / 2);
        const int  thread_index_in_subblock = thread_index_in_block % (subblock_size / 2);

        const int  lower_array_index =   block_index * constants.block_size
                                       + subblock_index * subblock_size
                                       + thread_index_in_subblock;

        if ( thread_position_in_grid < (constants.total_num_elements / 2) ) {
            const int  upper_array_index = lower_array_index + swap_span;

            if(    ( ( is_upward) && ( target_array[lower_array_index] > target_array[upper_array_index] ) )
                || ( (!is_upward) && ( target_array[lower_array_index] < target_array[upper_array_index] ) ) ) {

                    const int saved                 = target_array[lower_array_index];
                    target_array[lower_array_index] = target_array[upper_array_index];
                    target_array[upper_array_index] = saved;
            }
        }

        threadgroup_barrier( mem_flags::mem_device );
    }
}


kernel void bitonic_sort_one_swap_span_float(
    device       float*                   target_array             [[ buffer( 0) ]],
    device const bitonic_sort_constants&  constants                [[ buffer( 1) ]],
    const        uint                     thread_position_in_grid  [[ thread_position_in_grid ]],
    const        uint                     threadgroups_per_grid    [[ threadgroups_per_grid ]]
) {
    if ( thread_position_in_grid < (constants.total_num_elements / 2) ) {

        const int  half_block_size          = constants.block_size / 2;
        const int  block_index              = thread_position_in_grid / half_block_size;
        const int  thread_index_in_block    = thread_position_in_grid % half_block_size;

        const bool is_upward                = block_index % 2 == 0; // 0, 2, 4, 6,...

        const int  subblock_size            = constants.swap_span * 2;
        //const int  num_swaps_per_subblock   = constants.block_size / subblock_size;
        const int  subblock_index           = thread_index_in_block / (subblock_size / 2);
        const int  thread_index_in_subblock = thread_index_in_block % (subblock_size / 2);

        const int  lower_array_index =   block_index * constants.block_size
                                       + subblock_index * subblock_size
                                       + thread_index_in_subblock;

        const int  upper_array_index = lower_array_index + constants.swap_span;

        if(    ( ( is_upward) && ( target_array[lower_array_index] > target_array[upper_array_index] ) )
            || ( (!is_upward) && ( target_array[lower_array_index] < target_array[upper_array_index] ) ) ) {

            const float saved               = target_array[lower_array_index];
            target_array[lower_array_index] = target_array[upper_array_index];
            target_array[upper_array_index] = saved;
        }
    }
}


kernel void bitonic_sort_multiple_swap_spans_down_to_2_float(

    device       float*                   target_array             [[ buffer( 0) ]],
    device const bitonic_sort_constants&  constants                [[ buffer( 1) ]],
    const        uint                     thread_position_in_grid  [[ thread_position_in_grid ]],
    const        uint                     threadgroups_per_grid    [[ threadgroups_per_grid ]]
) {

    const int  half_block_size          = constants.block_size / 2;
    const int  block_index              = thread_position_in_grid / half_block_size;
    const int  thread_index_in_block    = thread_position_in_grid % half_block_size;

    const bool is_upward                = block_index % 2 == 0; // 0, 2, 4, 6,...

    for ( int swap_span = constants.swap_span; swap_span >= 1; swap_span /= 2 ) {

        const int  subblock_size            = swap_span * 2;
        //const int  num_swaps_per_subblock   = constants.block_size / subblock_size;
        const int  subblock_index           = thread_index_in_block / (subblock_size / 2);
        const int  thread_index_in_subblock = thread_index_in_block % (subblock_size / 2);

        const int  lower_array_index =   block_index * constants.block_size
                                       + subblock_index * subblock_size
                                       + thread_index_in_subblock;

        if ( thread_position_in_grid < (constants.total_num_elements / 2) ) {
            const int  upper_array_index = lower_array_index + swap_span;

            if(    ( ( is_upward) && ( target_array[lower_array_index] > target_array[upper_array_index] ) )
                || ( (!is_upward) && ( target_array[lower_array_index] < target_array[upper_array_index] ) ) ) {

                    const float saved               = target_array[lower_array_index];
                    target_array[lower_array_index] = target_array[upper_array_index];
                    target_array[upper_array_index] = saved;
            }
        }

        threadgroup_barrier( mem_flags::mem_device );
    }
}

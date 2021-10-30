#include <metal_stdlib>

using namespace metal;


struct parallel_order_checking_constants
{
    uint  total_num_elements;
};


// strategy
// --------
// 1. within-simdgroup comparisons
//    - coalesced read from the input and simd_shuffle & add result to atomic int.
//    - store boundary values to shared mem for step 2.
//
// 2. simdgroup-boundary comparisons per threadgroup
//    - coalesced read from the shared mem and simd_shuffle & add result to atomic int.
//    - store boundary values to shared mem for step 3.
//
// 3. threadgroup-boundary comparisons for the grid.
//    - coalesced read from the shared mem and simd_shuffle & add result to atomic int.

kernel void is_sorted_within_threadgroups(

    device const int*            in                               [[ buffer(0) ]],

    device       atomic_int*     is_unsorted                      [[ buffer(1) ]], // must be initialized to 0 before start

    device       int*            threadgroup_boundaries_prev_last [[ buffer(2) ]], // indexed in threadgroup_position_in_grid

    device       int*            threadgroup_boundaries_first     [[ buffer(3) ]], // indexed in threadgroup_position_in_grid - 1
                                                                                   // to enable coalesced read later in 
                                                                                   // is_sorted_at_threadgroup_boundaries(
    device const parallel_order_checking_constants& 
                                 constants                        [[ buffer(4) ]],

    const        uint            thread_position_in_grid          [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup   [[ thread_position_in_threadgroup ]],

    const        uint            threadgroup_position_in_grid     [[ threadgroup_position_in_grid ]],

    const        uint            threadgroups_per_grid            [[ threadgroups_per_grid ]],

    const        uint            thread_index_in_simdgroup        [[ thread_index_in_simdgroup ]],

    const        uint            simdgroup_index_in_threadgroup   [[ simdgroup_index_in_threadgroup ]],

    const        uint            simdgroups_per_threadgroup       [[ simdgroups_per_threadgroup ]]

) {

    threadgroup int simdgroup_boundary_prev_last[32];
    threadgroup int simdgroup_boundary_first[32];

    // is sorted within each simdgroup
    thread const int val =   (thread_position_in_grid < constants.total_num_elements)
                           ? in[ thread_position_in_grid ]
                           : INT_MAX
                           ;

    thread const int prev_val = simd_shuffle_up( val, 1);

    thread const int unsorted = ( prev_val > val ) ? 1 : 0 ;
    thread const int unsorted_sum = simd_sum(unsorted);

    if ( thread_index_in_simdgroup == 0 ) {

        if ( unsorted_sum > 0 ) {
            atomic_store_explicit( is_unsorted, 1, memory_order_relaxed );
        }

        if ( simdgroup_index_in_threadgroup == 0 ) {

            simdgroup_boundary_first[ simdgroups_per_threadgroup - 1 ] = INT_MAX;
        }
        else {
            simdgroup_boundary_first[ simdgroup_index_in_threadgroup - 1 ] = val;
        }
    }

    if ( thread_index_in_simdgroup == 31 ) {

        simdgroup_boundary_prev_last [ simdgroup_index_in_threadgroup ] = val;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // is sorted at the boundaries of simdgroups
    if ( simdgroup_index_in_threadgroup == 0 ) {

        thread const int val_prev =   (thread_index_in_simdgroup > 0)
                                     ? simdgroup_boundary_prev_last[thread_index_in_simdgroup - 1]
                                     : INT_MIN
                                     ;

        thread const int val_next = simdgroup_boundary_first[ thread_index_in_simdgroup ];

        thread const int unsorted = (val_prev > val_next)?1:0;

        thread const int unsorted_sum = simd_sum( unsorted );

        if ( unsorted_sum > 0 ) {
            if ( thread_index_in_simdgroup == 0 ) {

                atomic_store_explicit( is_unsorted, 1, memory_order_relaxed );
            }
        }
    }    

    // store the values of the  boundaries at the threadgroups

    if ( thread_position_in_threadgroup == 0 ) {

        if ( threadgroup_position_in_grid == 0 ) {
            threadgroup_boundaries_first[ threadgroups_per_grid - 1 ] = INT_MAX;
        }
        else {
            threadgroup_boundaries_first[ threadgroup_position_in_grid - 1] = val;
        }
    }
    
    if ( thread_position_in_threadgroup == 1023 ) {

        threadgroup_boundaries_prev_last[ threadgroup_position_in_grid ] = val;
    }
}


kernel void are_all_less_than_equal(

    device const int*            array1                    [[ buffer(0) ]],

    device const int*            array2                    [[ buffer(1) ]],

    device       atomic_int*     is_unsorted               [[ buffer(2) ]],

    device const parallel_order_checking_constants&
                                 constants                 [[ buffer(3) ]],

    const        uint            thread_position_in_grid   [[ thread_position_in_grid ]],

    const        uint            thread_index_in_simdgroup [[ thread_index_in_simdgroup ]]

) {

    thread const int val1 =   ( thread_position_in_grid < constants.total_num_elements )
                            ? array1[ thread_position_in_grid ]
                            : INT_MIN
                            ;

    thread const int val2 =   ( thread_position_in_grid < constants.total_num_elements )
                            ? array2[ thread_position_in_grid ]
                            : INT_MAX
                            ;

    thread const int unsorted = (val1 > val2) ? 1 : 0 ;

    thread const int unsorted_sum = simd_sum( unsorted );

    if ( thread_index_in_simdgroup == 0 ) {

        if ( unsorted_sum > 0 ) {
            atomic_store_explicit( is_unsorted, 1, memory_order_relaxed );
        }
    }
}

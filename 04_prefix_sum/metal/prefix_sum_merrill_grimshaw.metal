#include <metal_stdlib>

using namespace metal;

struct prefix_sum_constants
{
    uint  num_elements;
    uint  num_threads_per_partial_sum;
};


kernel void mg_get_partial_sums_32_X_int(

    device const int*            in                             [[ buffer(0) ]],

    device       int*            grid_prefix_sums               [[ buffer(1) ]],

    device const prefix_sum_constants& c                        [[ buffer(2) ]],

    threadgroup  int*            threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint            threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint            simdgroups_per_threadgroup     [[ simdgroups_per_threadgroup ]],

    const        uint            thread_execution_width         [[ thread_execution_width ]]

) {

    // Reset all the 32 elements in case threads_per_threadgroup < 1024.
    if ( simdgroup_index_in_threadgroup == 0 ) { 

        threadgroup_partial_sums[ thread_index_in_simdgroup ] = 0;
    }

    thread int local_sum_per_thread = 0;

    for (   size_t thread_position_in_grid_for_loop = thread_position_in_threadgroup + c.num_threads_per_partial_sum * threadgroup_position_in_grid

          ;    thread_position_in_grid_for_loop < c.num_threads_per_partial_sum * ( threadgroup_position_in_grid + 1 )
            && thread_position_in_grid_for_loop < c.num_elements

          ; thread_position_in_grid_for_loop += threads_per_threadgroup
    ) {

        local_sum_per_thread += in[ thread_position_in_grid_for_loop ];
    }

    thread const int warp_sum  = simd_sum( local_sum_per_thread );

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( thread_index_in_simdgroup == 0 ){

        threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] = warp_sum;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        thread const int local_sum2 = threadgroup_partial_sums[ thread_index_in_simdgroup ];
        thread const int warp_sum2  = simd_sum( local_sum2 );

        if ( thread_position_in_threadgroup == 0 ) {

            grid_prefix_sums[ threadgroup_position_in_grid ] = warp_sum2;
        }
    }
}


kernel void mg_scan_threadgroupwise_32_X_int(

    device       int*            inout                          [[ buffer(0) ]],

    device const prefix_sum_constants& c                        [[ buffer(1) ]],

    threadgroup  int*            threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint            threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint            simdgroups_per_threadgroup     [[ simdgroups_per_threadgroup ]],

    const        uint            thread_execution_width         [[ thread_execution_width ]]

) {
    // The SIMD group size must be 32.
    if ( thread_execution_width != 32 ) {
        return;
    }

    // Warp-wise prefix sum

    thread int local_sum = (thread_position_in_grid < c.num_elements) ? inout [ thread_position_in_grid ] : 0;

    if ( thread_index_in_simdgroup >=  1 ) local_sum += simd_shuffle_up( local_sum,  1 );
    if ( thread_index_in_simdgroup >=  2 ) local_sum += simd_shuffle_up( local_sum,  2 );
    if ( thread_index_in_simdgroup >=  4 ) local_sum += simd_shuffle_up( local_sum,  4 );
    if ( thread_index_in_simdgroup >=  8 ) local_sum += simd_shuffle_up( local_sum,  8 );
    if ( thread_index_in_simdgroup >= 16 ) local_sum += simd_shuffle_up( local_sum, 16 );

    if ( thread_index_in_simdgroup == 31 ) {

        if ( simdgroup_index_in_threadgroup == 0 ) {

            threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] = 0;
        }
        if ( simdgroup_index_in_threadgroup < (simdgroups_per_threadgroup - 1)) {

            threadgroup_partial_sums[ simdgroup_index_in_threadgroup + 1 ] = local_sum;
        }
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // Threadgroup-wise coarse prefix sum

    if ( simdgroup_index_in_threadgroup == 0 ) {

        thread int local_sum_rep = threadgroup_partial_sums[ thread_position_in_threadgroup ];

        if ( thread_index_in_simdgroup >=  1 && simdgroups_per_threadgroup >=  2 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   1 );
        if ( thread_index_in_simdgroup >=  2 && simdgroups_per_threadgroup >=  4 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   2 );
        if ( thread_index_in_simdgroup >=  4 && simdgroups_per_threadgroup >=  8 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   4 );
        if ( thread_index_in_simdgroup >=  8 && simdgroups_per_threadgroup >= 16 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   8 );
        if ( thread_index_in_simdgroup >= 16 && simdgroups_per_threadgroup >= 32 ) local_sum_rep += simd_shuffle_up( local_sum_rep,  16 );

        threadgroup_partial_sums[ thread_position_in_threadgroup ] = local_sum_rep;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // Propagate coarse prefix sum to warp-wise prefix sum
    if ( thread_position_in_grid < c.num_elements ) {

        inout[ thread_position_in_grid ] = local_sum + threadgroup_partial_sums[ simdgroup_index_in_threadgroup ];
    }
}



kernel void mg_scan_final_32_X_int(

    device const int*            in                             [[ buffer(0) ]],

    device       int*            out                            [[ buffer(1) ]],

    device const int*            grid_prefix_sums               [[ buffer(2) ]],

    device const prefix_sum_constants& c                        [[ buffer(3) ]],

    threadgroup  int*            threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint            simdgroups_per_threadgroup     [[ simdgroups_per_threadgroup ]],

    const        uint            thread_execution_width         [[ thread_execution_width ]]

) {
    threadgroup  int  base_for_thread_0;

    if ( thread_execution_width != 32 ) {
        return;
    }

    // Reset all the 32 elements in case threads_per_threadgroup < 1024.
    if ( simdgroup_index_in_threadgroup == 0 ) { 

        threadgroup_partial_sums[ thread_index_in_simdgroup ] = 0;
    }

    base_for_thread_0 =   ( threadgroup_position_in_grid == 0 )
                        ? 0
                        : grid_prefix_sums[ threadgroup_position_in_grid - 1 ];

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint thread_position_in_grid_for_loop = thread_position_in_threadgroup + c.num_threads_per_partial_sum * threadgroup_position_in_grid

          ;    thread_position_in_grid_for_loop < c.num_threads_per_partial_sum * ( threadgroup_position_in_grid + 1 )
            && thread_position_in_grid_for_loop < c.num_elements

          ; thread_position_in_grid_for_loop += threads_per_threadgroup
    ) {

        thread int local_sum =   in[ thread_position_in_grid_for_loop ];

        threadgroup_barrier( mem_flags::mem_threadgroup );

        if ( thread_position_in_threadgroup == 0) {
            local_sum += base_for_thread_0;
        }

        if ( thread_index_in_simdgroup >=  1 ) local_sum += simd_shuffle_up( local_sum,  1 );
        if ( thread_index_in_simdgroup >=  2 ) local_sum += simd_shuffle_up( local_sum,  2 );
        if ( thread_index_in_simdgroup >=  4 ) local_sum += simd_shuffle_up( local_sum,  4 );
        if ( thread_index_in_simdgroup >=  8 ) local_sum += simd_shuffle_up( local_sum,  8 );
        if ( thread_index_in_simdgroup >= 16 ) local_sum += simd_shuffle_up( local_sum, 16 );

        if ( thread_index_in_simdgroup == 31 ) {

            if ( simdgroup_index_in_threadgroup == 0 ) {

                threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] = 0;
            }
            if ( simdgroup_index_in_threadgroup < (simdgroups_per_threadgroup - 1) ) {

                threadgroup_partial_sums[ simdgroup_index_in_threadgroup + 1 ] = local_sum;
            }
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // Threadgroup-wise coarse prefix sum

        if ( simdgroup_index_in_threadgroup == 0 ) {

            thread int local_sum_rep = threadgroup_partial_sums[ thread_position_in_threadgroup ];

            if ( thread_index_in_simdgroup >=  1 && simdgroups_per_threadgroup >=  2 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   1 );
            if ( thread_index_in_simdgroup >=  2 && simdgroups_per_threadgroup >=  4 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   2 );
            if ( thread_index_in_simdgroup >=  4 && simdgroups_per_threadgroup >=  8 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   4 );
            if ( thread_index_in_simdgroup >=  8 && simdgroups_per_threadgroup >= 16 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   8 );
            if ( thread_index_in_simdgroup >= 16 && simdgroups_per_threadgroup >= 32 ) local_sum_rep += simd_shuffle_up( local_sum_rep,  16 );

            threadgroup_partial_sums[ thread_position_in_threadgroup ] = local_sum_rep;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // Propagate coarse prefix sum to warp-wise prefix sum
        thread int simd_group_base = threadgroup_partial_sums[ simdgroup_index_in_threadgroup ];

        out[ thread_position_in_grid_for_loop ] = local_sum + simd_group_base;

        if (thread_position_in_threadgroup == threads_per_threadgroup  - 1) {
            base_for_thread_0 = local_sum + simd_group_base;
        }
    }
}



kernel void mg_get_partial_sums_32_X_float(

    device const float*          in                             [[ buffer(0) ]],

    device       float*          grid_prefix_sums               [[ buffer(1) ]],

    device const prefix_sum_constants& c                        [[ buffer(2) ]],

    threadgroup  float*          threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]]
) {
    // Not implemented due to lack of simd instructions for float.
}


kernel void mg_scan_threadgroupwise_32_X_float(

    device       float*          inout                          [[ buffer(0) ]],

    device const prefix_sum_constants& c                        [[ buffer(1) ]],

    threadgroup  float*          threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]]

) {
    // Not implemented due to lack of simd instructions for float.
}



kernel void mg_scan_final_32_X_float(

    device const float*          in                             [[ buffer(0) ]],

    device       float*          out                            [[ buffer(1) ]],

    device const float*          grid_prefix_sums               [[ buffer(2) ]],

    device const prefix_sum_constants& c                        [[ buffer(3) ]],

    threadgroup  float*          threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]]
) {
    // Not implemented due to lack of simd instructions for float.
}

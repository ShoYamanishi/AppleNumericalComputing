#include <metal_stdlib>

using namespace metal;

struct prefix_sum_constants
{
    uint  num_elements;
};


kernel void scan_threadgroupwise_intermediate_32_32_int(

    device const int*            in                             [[ buffer(0) ]],

    device       int*            out                            [[ buffer(1) ]],

    device       int*            grid_prefix_sums               [[ buffer(2) ]],

    device const prefix_sum_constants& c                        [[ buffer(3) ]],

    threadgroup  int*            threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint            threadgroups_per_grid          [[ threadgroups_per_grid ]]
) {

    // Warp-wise prefix sum

    if ( thread_position_in_grid < c.num_elements ) {

        thread int local_sum = in [ thread_position_in_grid ];

        if ( thread_index_in_simdgroup >=  1 ) local_sum += simd_shuffle_up( local_sum,  1 );
        if ( thread_index_in_simdgroup >=  2 ) local_sum += simd_shuffle_up( local_sum,  2 );
        if ( thread_index_in_simdgroup >=  4 ) local_sum += simd_shuffle_up( local_sum,  4 );
        if ( thread_index_in_simdgroup >=  8 ) local_sum += simd_shuffle_up( local_sum,  8 );
        if ( thread_index_in_simdgroup >= 16 ) local_sum += simd_shuffle_up( local_sum, 16 );

        if ( thread_index_in_simdgroup == 31 ) {

            if ( simdgroup_index_in_threadgroup == 0 ) {

                threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] = 0;
            }
            if ( simdgroup_index_in_threadgroup < 31 ) {

                threadgroup_partial_sums[ simdgroup_index_in_threadgroup + 1 ] = local_sum;
            }
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // Threadgroup-wise coarse prefix sum

        if ( simdgroup_index_in_threadgroup == 0 ) {

            thread int local_sum_rep = threadgroup_partial_sums[ thread_position_in_threadgroup ];

            if ( thread_index_in_simdgroup >=  1 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   1 );
            if ( thread_index_in_simdgroup >=  2 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   2 );
            if ( thread_index_in_simdgroup >=  4 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   4 );
            if ( thread_index_in_simdgroup >=  8 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   8 );
            if ( thread_index_in_simdgroup >= 16 ) local_sum_rep += simd_shuffle_up( local_sum_rep,  16 );

            threadgroup_partial_sums[ thread_position_in_threadgroup ] = local_sum_rep;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // Propagate coarse prefix sum to warp-wise prefix sum

        out[ thread_position_in_grid ] =  ( simdgroup_index_in_threadgroup > 0 )
                                         ?( local_sum + threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] )
                                         :( local_sum );

        if ( threadgroups_per_grid > 1 ) {

            if ( thread_position_in_threadgroup == 1023 ) {

                if ( threadgroup_position_in_grid == 0 ) {
                    grid_prefix_sums[ 0 ] = 0;
                }

                if ( threadgroup_position_in_grid < threadgroups_per_grid  - 1 ) {

                    grid_prefix_sums[ threadgroup_position_in_grid + 1 ]
                        = local_sum + threadgroup_partial_sums[ simdgroup_index_in_threadgroup ];
                }
            }
        }
    }
}


kernel void add_base_32_32_int(

    device       int*            out                            [[ buffer(0) ]],

    device const int*            grid_prefix_sums               [[ buffer(1) ]],

    device const prefix_sum_constants& c                        [[ buffer(2) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]
) {
    if ( threadgroup_position_in_grid > 0 && thread_position_in_grid < c.num_elements ) {

        out[ thread_position_in_grid ] += grid_prefix_sums[ threadgroup_position_in_grid ];
    }
}


// threads per threadgroup must be 1024.
kernel void sum_threadgroup_32_32_int(

    device const int*            in                             [[ buffer(0) ]],

    device       int*            grid_prefix_sums               [[ buffer(1) ]],

    device const prefix_sum_constants& c                        [[ buffer(2) ]],

    threadgroup  int*            threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint            threadgroups_per_grid          [[ threadgroups_per_grid ]]
) {
    if ( thread_position_in_grid < c.num_elements ) {

        // Reset all the 32 elements in case threads_per_threadgroup < 1024.
        if ( simdgroup_index_in_threadgroup == 0 ) { 
            threadgroup_partial_sums[thread_index_in_simdgroup] = 0;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        thread const int local_sum = in [ thread_position_in_grid ];
        thread const int warp_sum  = simd_sum(local_sum);

        if ( thread_index_in_simdgroup == 0 ){
            threadgroup_partial_sums[simdgroup_index_in_threadgroup] = warp_sum;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        if ( simdgroup_index_in_threadgroup == 0 ) {

            thread const int local_sum2 = threadgroup_partial_sums[ thread_index_in_simdgroup ];
            thread const int warp_sum2  = simd_sum(local_sum2);

            if ( thread_position_in_threadgroup == 0 ) {

                grid_prefix_sums[ threadgroup_position_in_grid ] = warp_sum2;
            }
        }
    }
}


kernel void scan_with_base_threadgroupwise_32_32_int(

    device const int*            in                             [[ buffer(0) ]],

    device       int*            out                            [[ buffer(1) ]],

    device       int*            grid_prefix_sums               [[ buffer(2) ]],

    device const prefix_sum_constants& c                        [[ buffer(3) ]],

    threadgroup  int*            threadgroup_partial_sums       [[ threadgroup(0) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint            simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]]

) {
    if ( thread_position_in_grid < c.num_elements ) {

        // Warp-wise prefix sum

        thread int local_sum =   in[ thread_position_in_grid ]

                               + (  ( ( thread_position_in_threadgroup == 0 ) && ( threadgroup_position_in_grid > 0 ) )
                                  ? ( grid_prefix_sums[ threadgroup_position_in_grid - 1 ] )
                                  : 0  );

        if ( thread_index_in_simdgroup >=  1 ) local_sum += simd_shuffle_up( local_sum,  1 );
        if ( thread_index_in_simdgroup >=  2 ) local_sum += simd_shuffle_up( local_sum,  2 );
        if ( thread_index_in_simdgroup >=  4 ) local_sum += simd_shuffle_up( local_sum,  4 );
        if ( thread_index_in_simdgroup >=  8 ) local_sum += simd_shuffle_up( local_sum,  8 );
        if ( thread_index_in_simdgroup >= 16 ) local_sum += simd_shuffle_up( local_sum, 16 );

        if ( thread_index_in_simdgroup == 31 ) {

            if ( simdgroup_index_in_threadgroup == 0 ) {

                threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] = 0;
            }
            if ( simdgroup_index_in_threadgroup < 31 ) {

                threadgroup_partial_sums[ simdgroup_index_in_threadgroup + 1 ] = local_sum;
            }
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // Threadgroup-wise coarse prefix sum

        if ( simdgroup_index_in_threadgroup == 0 ) {

            thread int local_sum_rep = threadgroup_partial_sums[ thread_position_in_threadgroup ];

            if ( thread_index_in_simdgroup >=  1 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   1 );
            if ( thread_index_in_simdgroup >=  2 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   2 );
            if ( thread_index_in_simdgroup >=  4 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   4 );
            if ( thread_index_in_simdgroup >=  8 ) local_sum_rep += simd_shuffle_up( local_sum_rep,   8 );
            if ( thread_index_in_simdgroup >= 16 ) local_sum_rep += simd_shuffle_up( local_sum_rep,  16 );

            threadgroup_partial_sums[ thread_position_in_threadgroup ] = local_sum_rep;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // Propagate coarse prefix sum to warp-wise prefix sum
        out[ thread_position_in_grid ] =  ( simdgroup_index_in_threadgroup > 0 )
                                         ?( local_sum + threadgroup_partial_sums[ simdgroup_index_in_threadgroup ] )
                                         :( local_sum );
    }
}


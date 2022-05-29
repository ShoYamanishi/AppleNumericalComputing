#include <metal_stdlib>

using namespace metal;


struct radix_sort_constants
{
    uint  total_num_elements;
    uint  bit_right_shift;
    bool  flip_msb;
    bool  for_float;
};


static inline void make_prefix_sum_simdgroup_wise_4lanes(
    thread unsigned short& s1,
    thread unsigned short& s2,
    thread unsigned short& s3,
    thread unsigned short& s4,
    const uint thread_index_in_simdgroup
) {

    if ( thread_index_in_simdgroup >=  1 ) {

        s1 += simd_shuffle_up( s1,  1 );
        s2 += simd_shuffle_up( s2,  1 );
        s3 += simd_shuffle_up( s3,  1 );
        s4 += simd_shuffle_up( s4,  1 );
    }

    if ( thread_index_in_simdgroup >=  2 ) {

        s1 += simd_shuffle_up( s1,  2 );
        s2 += simd_shuffle_up( s2,  2 );
        s3 += simd_shuffle_up( s3,  2 );
        s4 += simd_shuffle_up( s4,  2 );
    }

    if ( thread_index_in_simdgroup >=  4 ) {

        s1 += simd_shuffle_up( s1,  4 );
        s2 += simd_shuffle_up( s2,  4 );
        s3 += simd_shuffle_up( s3,  4 );
        s4 += simd_shuffle_up( s4,  4 );
    }

    if ( thread_index_in_simdgroup >=  8 ) {

        s1 += simd_shuffle_up( s1,  8 );
        s2 += simd_shuffle_up( s2,  8 );
        s3 += simd_shuffle_up( s3,  8 );
        s4 += simd_shuffle_up( s4,  8 );
    }

    if ( thread_index_in_simdgroup >= 16 ) {

        s1 += simd_shuffle_up( s1, 16 );
        s2 += simd_shuffle_up( s2, 16 );
        s3 += simd_shuffle_up( s3, 16 );
        s4 += simd_shuffle_up( s4, 16 );
    }
}


static inline void make_prefix_sum_threadgroup_wise_4lanes(

    threadgroup unsigned short*  simdgroup_sums0,
    threadgroup unsigned short*  simdgroup_sums1,
    threadgroup unsigned short*  simdgroup_sums2,
    threadgroup unsigned short*  simdgroup_sums3,

    threadgroup unsigned short*  lane_counts_lane0,
    threadgroup unsigned short*  lane_counts_lane1,
    threadgroup unsigned short*  lane_counts_lane2,
    threadgroup unsigned short*  lane_counts_lane3,

    const       uint             thread_position_in_threadgroup,
    const       uint             thread_index_in_simdgroup,
    const       uint             simdgroup_index_in_threadgroup,
    const       uint             simdgroups_per_threadgroup
) {

    // Reset all the 32 elements in case threads_per_threadgroup < 1024.
    if ( simdgroup_index_in_threadgroup == 0 ) {

        simdgroup_sums0[ thread_index_in_simdgroup ] = 0;
        simdgroup_sums1[ thread_index_in_simdgroup ] = 0;
        simdgroup_sums2[ thread_index_in_simdgroup ] = 0;
        simdgroup_sums3[ thread_index_in_simdgroup ] = 0;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // within-simdgroup prefix sum

    thread unsigned short local_sum0 = lane_counts_lane0[ thread_position_in_threadgroup ];
    thread unsigned short local_sum1 = lane_counts_lane1[ thread_position_in_threadgroup ];
    thread unsigned short local_sum2 = lane_counts_lane2[ thread_position_in_threadgroup ];
    thread unsigned short local_sum3 = lane_counts_lane3[ thread_position_in_threadgroup ];

    make_prefix_sum_simdgroup_wise_4lanes( local_sum0,  local_sum1,  local_sum2,  local_sum3, thread_index_in_simdgroup );

    // in-threadgroup per-simdgroup prefix sum

    // NOTE: store the sum of the previous simdgroup to 
    //       simdgroup_sumsX's index to make the prefix-sum
    //       start with zero, i.e., make it exclusive-scan.

    if ( thread_index_in_simdgroup == 31 ) {

        if ( simdgroup_index_in_threadgroup == 0 ) {

            simdgroup_sums0[ simdgroup_index_in_threadgroup ] = 0;
            simdgroup_sums1[ simdgroup_index_in_threadgroup ] = 0;
            simdgroup_sums2[ simdgroup_index_in_threadgroup ] = 0;
            simdgroup_sums3[ simdgroup_index_in_threadgroup ] = 0;
        }
        if ( simdgroup_index_in_threadgroup < simdgroups_per_threadgroup - 1 ) {

            simdgroup_sums0[ simdgroup_index_in_threadgroup + 1 ] = local_sum0;
            simdgroup_sums1[ simdgroup_index_in_threadgroup + 1 ] = local_sum1;
            simdgroup_sums2[ simdgroup_index_in_threadgroup + 1 ] = local_sum2;
            simdgroup_sums3[ simdgroup_index_in_threadgroup + 1 ] = local_sum3;
        }
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        // use the simdgroup zero to make prefix-sum of simdgroups.

        thread unsigned short local_sum_sg0 = simdgroup_sums0[ thread_index_in_simdgroup ];
        thread unsigned short local_sum_sg1 = simdgroup_sums1[ thread_index_in_simdgroup ];
        thread unsigned short local_sum_sg2 = simdgroup_sums2[ thread_index_in_simdgroup ];
        thread unsigned short local_sum_sg3 = simdgroup_sums3[ thread_index_in_simdgroup ];

        make_prefix_sum_simdgroup_wise_4lanes( local_sum_sg0,  local_sum_sg1,  local_sum_sg2,  local_sum_sg3, thread_index_in_simdgroup );

        simdgroup_sums0[ thread_index_in_simdgroup ] = local_sum_sg0;
        simdgroup_sums1[ thread_index_in_simdgroup ] = local_sum_sg1;
        simdgroup_sums2[ thread_index_in_simdgroup ] = local_sum_sg2;
        simdgroup_sums3[ thread_index_in_simdgroup ] = local_sum_sg3;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // threadgroup-wise per-thread prefix sum

    lane_counts_lane0[ thread_position_in_threadgroup ] = local_sum0 + simdgroup_sums0[ simdgroup_index_in_threadgroup ];
    lane_counts_lane1[ thread_position_in_threadgroup ] = local_sum1 + simdgroup_sums1[ simdgroup_index_in_threadgroup ];
    lane_counts_lane2[ thread_position_in_threadgroup ] = local_sum2 + simdgroup_sums2[ simdgroup_index_in_threadgroup ];
    lane_counts_lane3[ thread_position_in_threadgroup ] = local_sum3 + simdgroup_sums3[ simdgroup_index_in_threadgroup ];

    threadgroup_barrier( mem_flags::mem_threadgroup );
}


static inline thread unsigned short calc_lane(thread const int v, const device struct radix_sort_constants& c )
{
    const unsigned short second_bit_from_lst = ( c.flip_msb ) ? 0x2 : 0x0;

    return ( ( v >> c.bit_right_shift ) & 0x3 )^ second_bit_from_lst;
}


static inline void count_up_lane(

    threadgroup unsigned short* lane_counts_lane0, 
    threadgroup unsigned short* lane_counts_lane1, 
    threadgroup unsigned short* lane_counts_lane2, 
    threadgroup unsigned short* lane_counts_lane3,
    const       unsigned short  lane,

    const       uint            thread_position_in_threadgroup
) {

    switch( lane ) {

      case 0:
        lane_counts_lane0[ thread_position_in_threadgroup ] = 1;
        lane_counts_lane1[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane2[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane3[ thread_position_in_threadgroup ] = 0;
        break;

      case 1:
        lane_counts_lane0[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane1[ thread_position_in_threadgroup ] = 1;
        lane_counts_lane2[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane3[ thread_position_in_threadgroup ] = 0;
        break;

      case 2:
        lane_counts_lane0[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane1[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane2[ thread_position_in_threadgroup ] = 1;
        lane_counts_lane3[ thread_position_in_threadgroup ] = 0;
        break;

      case 3:
        lane_counts_lane0[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane1[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane2[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane3[ thread_position_in_threadgroup ] = 1;
        break;

      default:
        lane_counts_lane0[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane1[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane2[ thread_position_in_threadgroup ] = 0;
        lane_counts_lane3[ thread_position_in_threadgroup ] = 0;
        break;
    }
}


static inline thread unsigned short get_base_pos_of_lane_in_sorted_array(

    const threadgroup unsigned short* lane_counts_lane0,
    const threadgroup unsigned short* lane_counts_lane1,
    const threadgroup unsigned short* lane_counts_lane2,
    const threadgroup unsigned short* lane_counts_lane3,

    const thread      unsigned short  lane,

    const             uint            threads_per_threadgroup
) {
    unsigned short base;

    switch( lane ) {

      case 0:
        base = 0;
        break;

      case 1:
        base = lane_counts_lane0[ threads_per_threadgroup - 1 ];
        break;

      case 2:
        base = lane_counts_lane0[ threads_per_threadgroup - 1 ] + lane_counts_lane1[ threads_per_threadgroup - 1 ];
        break;

      case 3:
        base = lane_counts_lane0[ threads_per_threadgroup - 1 ] + lane_counts_lane1[ threads_per_threadgroup - 1 ] + lane_counts_lane2[ threads_per_threadgroup - 1 ];
        break;

      default:
        base = 0;

    }
    return base;
}


static inline thread unsigned short get_prefix_sum_for_this_thread(

    const threadgroup unsigned short* lane_counts_lane0,
    const threadgroup unsigned short* lane_counts_lane1,
    const threadgroup unsigned short* lane_counts_lane2,
    const threadgroup unsigned short* lane_counts_lane3,

    const thread      unsigned short  lane,

    const             uint            thread_position_in_threadgroup
) {
    switch (lane) {

      case 0:
        return   ( thread_position_in_threadgroup > 0 )
               ? ( lane_counts_lane0[ thread_position_in_threadgroup ] - 1 )
               : 0
               ;
        break;

      case 1:
        return   ( thread_position_in_threadgroup > 0 )
               ? ( lane_counts_lane1[ thread_position_in_threadgroup ] - 1 )
               : 0
               ;
        break;

      case 2:
        return   ( thread_position_in_threadgroup > 0 )
               ? ( lane_counts_lane2[ thread_position_in_threadgroup ] - 1 )
               : 0
               ;
        break;

      case 3:
        return   ( thread_position_in_threadgroup > 0 )
               ? ( lane_counts_lane3[ thread_position_in_threadgroup ] - 1 )
               : 0
               ;
        break;

      default:
        return 0;
    }
}


// algorithm 2 of Ha Krüger, Silva 2009
kernel void four_way_prefix_sum_with_inblock_shuffle(

    device       int*            target_array                               [[ buffer( 0) ]], // after the call target_array will be locally sorted per threadgroup.

    device       int*            partial_sums_per_threadgroup_lane0         [[ buffer( 1) ]],
    device       int*            partial_sums_per_threadgroup_lane1         [[ buffer( 2) ]],
    device       int*            partial_sums_per_threadgroup_lane2         [[ buffer( 3) ]],
    device       int*            partial_sums_per_threadgroup_lane3         [[ buffer( 4) ]],

    device       unsigned short* start_pos_within_threadgroups_lane1        [[ buffer( 5) ]],
    device       unsigned short* start_pos_within_threadgroups_lane2        [[ buffer( 6) ]],
    device       unsigned short* start_pos_within_threadgroups_lane3        [[ buffer( 7) ]],

    device const struct radix_sort_constants& 
                                 constants                                  [[ buffer( 8) ]],

    threadgroup  unsigned short* lane_counts_lane0                          [[ threadgroup(0) ]], // prefix-sum of occurrence counts per lane for this threadgroup is calculated.
    threadgroup  unsigned short* lane_counts_lane1                          [[ threadgroup(1) ]],
    threadgroup  unsigned short* lane_counts_lane2                          [[ threadgroup(2) ]],
    threadgroup  unsigned short* lane_counts_lane3                          [[ threadgroup(3) ]],
    
    threadgroup  int*            values_sorted                              [[ threadgroup(4) ]], // used to make the write-back to target_array coalesced.

    const        uint            thread_position_in_grid                    [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup             [[ thread_position_in_threadgroup ]],

    const        uint            threadgroup_position_in_grid               [[ threadgroup_position_in_grid ]],

    const        uint            threadgroups_per_grid                      [[ threadgroups_per_grid ]],

    const        uint            thread_index_in_simdgroup                  [[ thread_index_in_simdgroup ]],

    const        uint            simdgroup_index_in_threadgroup             [[ simdgroup_index_in_threadgroup ]],

    const        uint            threads_per_threadgroup                    [[ threads_per_threadgroup ]],

    const        uint            simdgroups_per_threadgroup                 [[ simdgroups_per_threadgroup ]],

    const        uint            thread_execution_width                     [[ thread_execution_width ]]
) {

    // scratch memory for make_prefix_sum_threadgroup_wise_4lanes
    threadgroup unsigned short simdgroup_sums0[ 32 ];
    threadgroup unsigned short simdgroup_sums1[ 32 ];
    threadgroup unsigned short simdgroup_sums2[ 32 ];
    threadgroup unsigned short simdgroup_sums3[ 32 ];

    // The SIMD group size must be 32.
    if ( thread_execution_width != 32 ) {
        return;
    }

    thread const bool is_valid_element = thread_position_in_grid < constants.total_num_elements;

    thread const int lane =   is_valid_element
                            ? calc_lane( target_array[ thread_position_in_grid ],  constants )
                            : 0xFF
                            ;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    count_up_lane( lane_counts_lane0, lane_counts_lane1, lane_counts_lane2, lane_counts_lane3, (unsigned short)lane, thread_position_in_threadgroup );

    threadgroup_barrier( mem_flags::mem_threadgroup );

    make_prefix_sum_threadgroup_wise_4lanes(

        simdgroup_sums0,   simdgroup_sums1,   simdgroup_sums2,   simdgroup_sums3,
        lane_counts_lane0, lane_counts_lane1, lane_counts_lane2, lane_counts_lane3, 
        thread_position_in_threadgroup, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, simdgroups_per_threadgroup
    );

    if ( is_valid_element ) {

        unsigned short dist_base   = get_base_pos_of_lane_in_sorted_array( lane_counts_lane0, lane_counts_lane1, lane_counts_lane2, lane_counts_lane3, lane, threads_per_threadgroup ); 

        unsigned short dist_offset = get_prefix_sum_for_this_thread( lane_counts_lane0, lane_counts_lane1, lane_counts_lane2, lane_counts_lane3, lane, thread_position_in_threadgroup  );

        values_sorted [ dist_base + dist_offset ] = target_array[ thread_position_in_grid ];
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( is_valid_element ) {      

        // coalesced write
        target_array[ thread_position_in_grid ] = values_sorted[ thread_position_in_threadgroup ];
    }

    if( thread_position_in_threadgroup == 0 ) {

        const unsigned short sum_lane0 = lane_counts_lane0[ threads_per_threadgroup - 1 ];
        const unsigned short sum_lane1 = lane_counts_lane1[ threads_per_threadgroup - 1 ];
        const unsigned short sum_lane2 = lane_counts_lane2[ threads_per_threadgroup - 1 ];
        const unsigned short sum_lane3 = lane_counts_lane3[ threads_per_threadgroup - 1 ];

        start_pos_within_threadgroups_lane1[threadgroup_position_in_grid ] = sum_lane0;                         // start pos of lane 1
        start_pos_within_threadgroups_lane2[threadgroup_position_in_grid ] = sum_lane0 + sum_lane1;             // start pos of lane 2
        start_pos_within_threadgroups_lane3[threadgroup_position_in_grid ] = sum_lane0 + sum_lane1 + sum_lane2; // start pos of lane 3

        partial_sums_per_threadgroup_lane0[ threadgroup_position_in_grid ] = sum_lane0;
        partial_sums_per_threadgroup_lane1[ threadgroup_position_in_grid ] = sum_lane1;
        partial_sums_per_threadgroup_lane2[ threadgroup_position_in_grid ] = sum_lane2;
        partial_sums_per_threadgroup_lane3[ threadgroup_position_in_grid ] = sum_lane3;
    }
}


// based on algorithm 4 of Ha Krüger, Silva 2009
kernel void coalesced_block_mapping_for_the_n_chunk_input(

    device const int*            src_array_sorted_within_threadgroups       [[ buffer(0) ]],

    device       int*            target_array                               [[ buffer(1) ]],

    // they start with the first element, i.e., inclusive-scan.
    device       int*            partial_sums_per_threadgroup_lane0         [[ buffer(2) ]],
    device       int*            partial_sums_per_threadgroup_lane1         [[ buffer(3) ]],
    device       int*            partial_sums_per_threadgroup_lane2         [[ buffer(4) ]],
    device       int*            partial_sums_per_threadgroup_lane3         [[ buffer(5) ]],

    device       unsigned short* start_pos_within_threadgroups_lane1        [[ buffer(6) ]],
    device       unsigned short* start_pos_within_threadgroups_lane2        [[ buffer(7) ]],
    device       unsigned short* start_pos_within_threadgroups_lane3        [[ buffer(8) ]],

    device const struct radix_sort_constants& 
                                 constants                                  [[ buffer(9) ]],

    // used to make the write-back to target_array coalesced.
    threadgroup  int*            copy_src_array                             [[ threadgroup(0) ]],

    const        uint            thread_position_in_threadgroup             [[ thread_position_in_threadgroup ]],

    const        uint            thread_position_in_grid                    [[ thread_position_in_grid ]],

    const        uint            threadgroup_position_in_grid               [[ threadgroup_position_in_grid ]],

    const        uint            threadgroups_per_grid                      [[ threadgroups_per_grid ]],

    const        uint            threads_per_threadgroup                    [[ threads_per_threadgroup ]],

    const        uint            simdgroups_per_threadgroup                 [[ simdgroups_per_threadgroup ]],

    const        uint            thread_execution_width                     [[ thread_execution_width ]]
) {


    threadgroup uint           lane_start_dst[4];
    threadgroup unsigned short lane_start_src_in_threadgroup[5]; // 5th element is sentinel.

    if ( thread_position_in_threadgroup == 0 ) {

        const int total_sum_lane0 = partial_sums_per_threadgroup_lane0[ threadgroups_per_grid - 1 ];
        const int total_sum_lane1 = partial_sums_per_threadgroup_lane1[ threadgroups_per_grid - 1 ];
        const int total_sum_lane2 = partial_sums_per_threadgroup_lane2[ threadgroups_per_grid - 1 ];

        lane_start_dst[0] = 0;
        lane_start_dst[1] = total_sum_lane0;
        lane_start_dst[2] = total_sum_lane0 + total_sum_lane1;
        lane_start_dst[3] = total_sum_lane0 + total_sum_lane1 + total_sum_lane2;

        if ( threadgroup_position_in_grid > 0 ) {
                                 
            lane_start_dst[0] += partial_sums_per_threadgroup_lane0[ threadgroup_position_in_grid - 1 ];
            lane_start_dst[1] += partial_sums_per_threadgroup_lane1[ threadgroup_position_in_grid - 1 ];
            lane_start_dst[2] += partial_sums_per_threadgroup_lane2[ threadgroup_position_in_grid - 1 ];
            lane_start_dst[3] += partial_sums_per_threadgroup_lane3[ threadgroup_position_in_grid - 1 ];
        }

        lane_start_src_in_threadgroup[0] = 0;
        lane_start_src_in_threadgroup[1] = start_pos_within_threadgroups_lane1[ threadgroup_position_in_grid ];
        lane_start_src_in_threadgroup[2] = start_pos_within_threadgroups_lane2[ threadgroup_position_in_grid ];
        lane_start_src_in_threadgroup[3] = start_pos_within_threadgroups_lane3[ threadgroup_position_in_grid ];
        lane_start_src_in_threadgroup[4] = threads_per_threadgroup; // sentinel
    }

    if ( thread_position_in_grid < constants.total_num_elements ) {

        copy_src_array[ thread_position_in_threadgroup ] = src_array_sorted_within_threadgroups[ thread_position_in_grid ];
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // coalesced write to target array from copy_src_array.


    if ( constants.bit_right_shift == 30 && constants.for_float ) {

        const int lane2_start_dst =   partial_sums_per_threadgroup_lane0[ threadgroups_per_grid - 1 ]
                                    + partial_sums_per_threadgroup_lane1[ threadgroups_per_grid - 1 ];

        // Should unroll manually if the compiler doesn't?
        for ( int lane = 0; lane < 2 ; lane++ ) {

            const int _lane_start_dst = lane2_start_dst - lane_start_dst[lane] - 1;

            const int simd_group_align_offset = _lane_start_dst % 32;
            const int index_src = ( thread_position_in_threadgroup + simd_group_align_offset ) % threads_per_threadgroup;

            const bool valid_src =    ( threadgroup_position_in_grid < (threadgroups_per_grid - 1)  )
                                   || ( index_src < (int)(constants.total_num_elements % threads_per_threadgroup) )
                                   || ( (constants.total_num_elements % threads_per_threadgroup) == 0 )
                               ;
            if ( valid_src ) {

                const int value_src  = copy_src_array[ index_src ];

                const int lane_src   = calc_lane( value_src, constants );
    
                if ( lane_src == lane ) {

                    const int lane_begin = lane_start_src_in_threadgroup[ lane     ];

                    const int lane_end   = lane_start_src_in_threadgroup[ lane + 1 ];

                    if ( lane_begin <= index_src && index_src < lane_end ) {

                        const int offset_from_lane_start = index_src - lane_begin;

                        target_array [ _lane_start_dst - offset_from_lane_start ]  = value_src;
                    }
                }
            }
        }
    }
    else {

        // Should unroll manually if the compiler dosn't?
        for ( int lane = 0; lane < 2 ; lane++ ) {

            const int _lane_start_dst         = lane_start_dst[lane];

            const int simd_group_align_offset = _lane_start_dst % 32;

            const int index_src_candidate = thread_position_in_threadgroup - simd_group_align_offset;

            const int index_src =   (index_src_candidate >=0) 
                                  ? index_src_candidate 
                                  : index_src_candidate + threads_per_threadgroup
                                  ;

            const bool valid_src =    ( threadgroup_position_in_grid < (threadgroups_per_grid - 1)  )
                                   || ( index_src < (int)(constants.total_num_elements % threads_per_threadgroup) )
                                   || ( (constants.total_num_elements % threads_per_threadgroup) == 0 )
                                   ;
            if ( valid_src ) {

                const int value_src  = copy_src_array[ index_src ];

                const int lane_src   = calc_lane( value_src, constants );

                if ( lane_src == lane ) {

                    const int lane_begin = lane_start_src_in_threadgroup[ lane     ];

                    const int lane_end   = lane_start_src_in_threadgroup[ lane + 1 ];

                    if ( lane_begin <= index_src && index_src < lane_end ) {

                        const int offset_from_lane_start = index_src - lane_begin;

                        target_array [ _lane_start_dst + offset_from_lane_start ]  = value_src;
                    }
                }
            }
        }
    }

    for ( int lane = 2; lane < 4 ; lane++ ) {

        const int _lane_start_dst         = lane_start_dst[lane];

        const int simd_group_align_offset = _lane_start_dst % 32;

        const int index_src_candidate = thread_position_in_threadgroup - simd_group_align_offset;

        const int index_src =   (index_src_candidate >=0) 
                              ? index_src_candidate 
                              : index_src_candidate + threads_per_threadgroup
                              ;

        const bool valid_src =    ( threadgroup_position_in_grid < (threadgroups_per_grid - 1)  )
                               || ( index_src < (int)(constants.total_num_elements % threads_per_threadgroup) )
                               || ( (constants.total_num_elements % threads_per_threadgroup) == 0 )
                               ;

        if ( valid_src ) {

            const int value_src  = copy_src_array[ index_src ];

            const int lane_src   = calc_lane( value_src, constants );

            if ( lane_src == lane ) {

                const int lane_begin = lane_start_src_in_threadgroup[ lane     ];

                const int lane_end   = lane_start_src_in_threadgroup[ lane + 1 ];

                if ( lane_begin <= index_src && index_src < lane_end ) {

                    const int offset_from_lane_start = index_src - lane_begin;

                    target_array [ _lane_start_dst + offset_from_lane_start ]  = value_src;
                }
            }
        }
    }
}


kernel void uncoalesced_block_mapping_for_the_n_chunk_input(

    device const int*            src_array_sorted_within_threadgroups       [[ buffer(0) ]],

    device       int*            target_array                               [[ buffer(1) ]],

    // they start with the first element, i.e., inclusive-scan.
    device       int*            partial_sums_per_threadgroup_lane0         [[ buffer(2) ]],
    device       int*            partial_sums_per_threadgroup_lane1         [[ buffer(3) ]],
    device       int*            partial_sums_per_threadgroup_lane2         [[ buffer(4) ]],
    device       int*            partial_sums_per_threadgroup_lane3         [[ buffer(5) ]],

    device       unsigned short* start_pos_within_threadgroups_lane1        [[ buffer(6) ]],
    device       unsigned short* start_pos_within_threadgroups_lane2        [[ buffer(7) ]],
    device       unsigned short* start_pos_within_threadgroups_lane3        [[ buffer(8) ]],

    device const struct radix_sort_constants& 
                                 constants                                  [[ buffer(9) ]],

    const        uint            thread_position_in_threadgroup             [[ thread_position_in_threadgroup ]],

    const        uint            thread_position_in_grid                    [[ thread_position_in_grid ]],

    const        uint            threadgroup_position_in_grid               [[ threadgroup_position_in_grid ]],

    const        uint            threadgroups_per_grid                      [[ threadgroups_per_grid ]]
) {

    threadgroup uint           lane_start_dst[4];
    threadgroup unsigned short lane_start_src_in_threadgroup[4];

    if ( thread_position_in_threadgroup == 0 ) {

        const int total_sum_lane0 = partial_sums_per_threadgroup_lane0[ threadgroups_per_grid - 1 ];
        const int total_sum_lane1 = partial_sums_per_threadgroup_lane1[ threadgroups_per_grid - 1 ];
        const int total_sum_lane2 = partial_sums_per_threadgroup_lane2[ threadgroups_per_grid - 1 ];

        lane_start_dst[0] = 0;
        lane_start_dst[1] = total_sum_lane0;
        lane_start_dst[2] = total_sum_lane0 + total_sum_lane1;
        lane_start_dst[3] = total_sum_lane0 + total_sum_lane1 + total_sum_lane2;

        if ( threadgroup_position_in_grid > 0 ) {
                                 
            lane_start_dst[0] += partial_sums_per_threadgroup_lane0[ threadgroup_position_in_grid - 1 ];
            lane_start_dst[1] += partial_sums_per_threadgroup_lane1[ threadgroup_position_in_grid - 1 ];
            lane_start_dst[2] += partial_sums_per_threadgroup_lane2[ threadgroup_position_in_grid - 1 ];
            lane_start_dst[3] += partial_sums_per_threadgroup_lane3[ threadgroup_position_in_grid - 1 ];
        }

        lane_start_src_in_threadgroup[0] = 0;
        lane_start_src_in_threadgroup[1] = start_pos_within_threadgroups_lane1[ threadgroup_position_in_grid ];
        lane_start_src_in_threadgroup[2] = start_pos_within_threadgroups_lane2[ threadgroup_position_in_grid ];
        lane_start_src_in_threadgroup[3] = start_pos_within_threadgroups_lane3[ threadgroup_position_in_grid ];
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( thread_position_in_grid < constants.total_num_elements ) {

        const int val =  src_array_sorted_within_threadgroups[ thread_position_in_grid ];
        const int lane = calc_lane( val, constants );

        const int src_offset = thread_position_in_threadgroup - lane_start_src_in_threadgroup[lane];

        if ( constants.bit_right_shift == 30 && constants.for_float && lane < 2 ) {

            const int lane2_start_dst =   partial_sums_per_threadgroup_lane0[ threadgroups_per_grid - 1 ]
                                        + partial_sums_per_threadgroup_lane1[ threadgroups_per_grid - 1 ];

            const int _lane_start_dst = lane2_start_dst - lane_start_dst[lane] - 1;

            target_array[ _lane_start_dst - src_offset ] = val;
        }
        else {
            target_array[ lane_start_dst[lane] + src_offset ] = val;
        }
    }

}


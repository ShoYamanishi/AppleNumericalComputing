#include <metal_stdlib>

using namespace metal;

struct sparse_matrix_vector_constants {
    uint  num_rows;
    uint  num_columns;
    uint  num_nnz;
    uint  num_blocks;
};

kernel void sparse_matrix_vector_row_per_thread (

    device const int*   const csr_row_ptrs                      [[ buffer(0) ]],
    device const int*   const csr_column_indices                [[ buffer(1) ]],
    device const float* const csr_values                        [[ buffer(2) ]],
    device const float* const csr_vector                        [[ buffer(3) ]],
    device float*             output_vector                     [[ buffer(4) ]],

    device const sparse_matrix_vector_constants&
                              constants                         [[ buffer(5) ]],

    const        uint         thread_position_in_grid           [[ thread_position_in_grid ]]
) {
    if ( thread_position_in_grid < constants.num_rows ) {

        const int pos_begin = csr_row_ptrs[thread_position_in_grid     ];
        const int pos_end   = csr_row_ptrs[thread_position_in_grid + 1 ];

        float sum = 0.0;
        for ( int pos = pos_begin; pos < pos_end; pos++ ) {

            sum += ( csr_values[pos] * csr_vector [ csr_column_indices[pos] ] );
        }

        output_vector [ thread_position_in_grid ] = sum;
    }
}



kernel void sparse_matrix_vector_adaptive (

    device const int*   const csr_block_ptrs                    [[ buffer(0) ]],
    device const int*   const csr_thrads_per_row                [[ buffer(1) ]],
    device const int*   const csr_max_iters                     [[ buffer(2) ]],
    device const int*   const csr_row_ptrs                      [[ buffer(3) ]],
    device const int*   const csr_column_indices                [[ buffer(4) ]],
    device const float* const csr_values                        [[ buffer(5) ]],
    device const float* const csr_vector                        [[ buffer(6) ]],
    device float*             output_vector                     [[ buffer(7) ]],

    device const sparse_matrix_vector_constants&
                              constants                         [[ buffer(8) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint            thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint            threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]]

) {

    threadgroup float cache[1024];

    const uint block_row_begin = csr_block_ptrs     [ threadgroup_position_in_grid     ];
    const uint block_row_end   = csr_block_ptrs     [ threadgroup_position_in_grid + 1 ];
    const uint threads_per_row = csr_thrads_per_row [ threadgroup_position_in_grid     ];
    const uint max_iters       = csr_max_iters      [ threadgroup_position_in_grid     ];

    const uint row_pos = thread_position_in_threadgroup / threads_per_row + block_row_begin;

    if ( row_pos < block_row_end ) {

        const uint offset  = thread_position_in_threadgroup % threads_per_row;

        const uint pos_begin = csr_row_ptrs[row_pos     ];
        const uint pos_end   = csr_row_ptrs[row_pos + 1 ];

        float sum = 0.0;
        for ( uint i = 0; i < max_iters ; i++ ) {
            const uint pos = pos_begin + offset + i * threads_per_row;
            if ( pos < pos_end ) {
                sum += ( csr_values[pos] * csr_vector [ csr_column_indices[pos] ] );
            }
        }

        cache[thread_position_in_threadgroup] = sum;

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // reduce    
        for ( uint i = threads_per_row/2 ; i > 0; i >>= 1 ) {
            if ( i>16 ) {
                if ( offset < i ){        
                    sum = cache[thread_position_in_threadgroup ] + cache[thread_position_in_threadgroup + i];
                    cache[thread_position_in_threadgroup] = sum;
                }
            }
            else {
                sum += simd_shuffle_down(sum, i);
            }           
            threadgroup_barrier( mem_flags::mem_threadgroup );
        }

        if (offset == 0) {
            output_vector[row_pos] = sum;
        }
    }
}

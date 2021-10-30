#include <metal_stdlib>

using namespace metal;

struct dense_matrix_vector_constants {
    int  M;
    int  N;
};


kernel void mult_col_major_threads_over_rows (

    device float*                    mat                            [[ buffer(0) ]],

    device float*                    vec                            [[ buffer(1) ]],

    device float*                    outvec                         [[ buffer(2) ]],

    device const dense_matrix_vector_constants& constants           [[ buffer(3) ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]

) {

    if ( (int)thread_position_in_grid < constants.M ) {


        float sum = 0.0;
        for ( int col = 0; col < constants.N; col++ ) {

            sum += ( mat[ thread_position_in_grid + constants.M * col ] * vec[ col ] );
        }

        outvec[ thread_position_in_grid ] = sum;
    }
}


kernel void mult_row_major_threads_over_rows (

    device float*                    mat                            [[ buffer(0) ]],

    device float*                    vec                            [[ buffer(1) ]],

    device float*                    outvec                         [[ buffer(2) ]],

    device const dense_matrix_vector_constants& constants           [[ buffer(3) ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]

) {
    if ( (int)thread_position_in_grid < constants.M ) {

        float sum = 0.0;
        const uint row_base = thread_position_in_grid * constants.N;

        for ( int col = 0; col < constants.N; col++ ) {

            sum += ( mat[ row_base + col ] * vec[ col ] );
        }

        outvec[ thread_position_in_grid ] = sum;
    }
}


kernel void mult_col_major_threads_over_columns (

    device float*                    mat                            [[ buffer(0) ]],

    device float*                    vec                            [[ buffer(1) ]],

    device float*                    outvec                         [[ buffer(2) ]],

    device const dense_matrix_vector_constants& constants           [[ buffer(3) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint                threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint                thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint                simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint                simdgroups_per_threadgroup     [[ simdgroups_per_threadgroup ]]
) {
    const int THREADS_PER_THREADGROUP      = 1024;    // macos

    threadgroup float sum_cache[ THREADS_PER_THREADGROUP ];

    const int row = threadgroup_position_in_grid;

    float sum = 0.0;
    for ( int col  = thread_position_in_threadgroup ; col < constants.N ; col += threads_per_threadgroup ) {
        sum += ( mat[ row + constants.M * col ] * vec[ col ] );
    }

    const float warp_sum = simd_sum (sum);

    if ( thread_index_in_simdgroup == 0 ){

        sum_cache[ simdgroup_index_in_threadgroup ] = warp_sum;
    }
    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        const float local_sum = (thread_index_in_simdgroup< simdgroups_per_threadgroup)? sum_cache[ thread_index_in_simdgroup ] : 0.0;

        const float warp_sum =  simd_sum( local_sum );

        if ( thread_position_in_threadgroup == 0 ) {
            outvec[row] =warp_sum;
        }
    }
}


kernel void mult_row_major_threads_over_columns (

    device float*                    mat                            [[ buffer(0) ]],

    device float*                    vec                            [[ buffer(1) ]],

    device float*                    outvec                         [[ buffer(2) ]],

    device const dense_matrix_vector_constants& constants           [[ buffer(3) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint                threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint                thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint                simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint                simdgroups_per_threadgroup     [[ simdgroups_per_threadgroup ]]
) {
    const int THREADS_PER_THREADGROUP      = 1024;    // macos

    threadgroup float sum_cache[ THREADS_PER_THREADGROUP ];

    const int row = threadgroup_position_in_grid;

    float sum = 0.0;
    for ( int col  = thread_position_in_threadgroup ; col < constants.N ; col += threads_per_threadgroup ) {
        sum += ( mat[ row * constants.N + col ] * vec[ col ] );
    }

    const float warp_sum = simd_sum (sum);

    if ( thread_index_in_simdgroup == 0 ){

        sum_cache[ simdgroup_index_in_threadgroup ] = warp_sum;
    }
    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        const float local_sum = (thread_index_in_simdgroup< simdgroups_per_threadgroup)? sum_cache[ thread_index_in_simdgroup ] : 0.0;

        const float warp_sum =  simd_sum( local_sum );

        if ( thread_position_in_threadgroup == 0 ) {
            outvec[row] =warp_sum;
        }
    }
}


// - input vector cached
// - col major for coalesced load
// - one thread per row
// NOTE: Use of threadgroup memory in metal does not make it faster if the num rows exceeds 1K.
kernel void mult_col_major_cache_vector(

    device float*                    mat                            [[ buffer(0) ]],

    device float*                    vec                            [[ buffer(1) ]],

    device float*                    outvec                         [[ buffer(2) ]],

    device const dense_matrix_vector_constants& constants           [[ buffer(3) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]

) {
    const int THREADS_PER_THREADGROUP      = 1024;    // macos
    const int THREADGROUP_MEM_MAX_IN_FLOAT = 8* 1024; // macos

    threadgroup float v_cache[ THREADGROUP_MEM_MAX_IN_FLOAT ];

    for ( int col_base = 0; col_base < constants.N; col_base += THREADGROUP_MEM_MAX_IN_FLOAT ) {

        for ( int row_base = 0; row_base < constants.M; row_base += THREADS_PER_THREADGROUP ){


            for ( int i = 0; i < THREADGROUP_MEM_MAX_IN_FLOAT / THREADS_PER_THREADGROUP; i++ ) {

                const int col_base_v_cache = i * THREADS_PER_THREADGROUP + thread_position_in_threadgroup;

                if ( col_base + col_base_v_cache < constants.N ) {

                    v_cache[ col_base_v_cache ] = vec[ col_base + col_base_v_cache ];
                }
            }

            threadgroup_barrier( mem_flags::mem_threadgroup );

            const int row = row_base + thread_position_in_threadgroup;

            if ( row < constants.M ) {

                const int col_end =   ( constants.N - col_base < THREADGROUP_MEM_MAX_IN_FLOAT )
                                    ? ( constants.N - col_base )
                                    : THREADGROUP_MEM_MAX_IN_FLOAT ;

                float sum = 0.0;

                for ( int k = 0; k < col_end; k++ ) {

                    sum += ( mat[ row + constants.M * ( col_base + k) ] * v_cache[k] );
//                    sum += ( mat[ row + constants.M * ( col_base + k) ] * vec[ col_base + k] );
                }

                outvec[row] += sum;
            }

            threadgroup_barrier( mem_flags::mem_device );
        }    
    }
}

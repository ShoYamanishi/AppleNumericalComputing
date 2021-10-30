#include <metal_stdlib>

using namespace metal;

struct jacobi_solver_constants {
    int  dim;
};

void atomic_add_float( device atomic_uint* atom_var, const float val )
{
    uint  fetched_uint,  assigning_uint;
    float fetched_float, assigning_float;

    fetched_uint = atomic_exchange_explicit( atom_var, 0, memory_order_relaxed );

    fetched_float = *( (thread float*) &fetched_uint );

    assigning_float = fetched_float + val;

    assigning_uint =  *( (thread uint*) &assigning_float );

    while ( (fetched_uint = atomic_exchange_explicit( atom_var, assigning_uint, memory_order_relaxed ) ) != 0 )  {

        uint fetched_uint_again = atomic_exchange_explicit( atom_var, 0, memory_order_relaxed );

        float fetched_float_again = *( (thread float*) &fetched_uint_again );

        fetched_float = *( (thread float*) &(fetched_uint) );

        assigning_float = fetched_float_again + fetched_float;

        assigning_uint =  *( (thread uint*) &assigning_float );
    }
}


kernel void solve_col_major (

    device const float*              A                              [[ buffer(0) ]],

    device const float*              Dinv                           [[ buffer(1) ]],

    device const float*              b                              [[ buffer(2) ]],

    device const float*              xin                            [[ buffer(3) ]],

    device float*                    xout                           [[ buffer(4) ]],

    device atomic_uint*              x_error                        [[ buffer(5) ]],

    device const jacobi_solver_constants& constants                 [[ buffer(6) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]

) {

    if ( (int)thread_position_in_grid < constants.dim ) {

        const int row = thread_position_in_grid;

        float sum = 0.0;

        for ( int col = 0; col < constants.dim; col++ ) {

            sum += ( A[ row + constants.dim * col] * xin[col] );
        }

        xout[row] = (b[row] - sum)*Dinv[row];

        atomic_add_float( x_error, (xout[row] - xin[row])*(xout[row] - xin[row]) );
    }
}



kernel void solve_row_major (

    device const float*              A                              [[ buffer(0) ]],

    device const float*              Dinv                           [[ buffer(1) ]],

    device const float*              b                              [[ buffer(2) ]],

    device const float*              xin                            [[ buffer(3) ]],

    device float*                    xout                           [[ buffer(4) ]],

    device atomic_uint*              x_error                        [[ buffer(5) ]],

    device const jacobi_solver_constants& constants                 [[ buffer(6) ]],

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
    for ( int col  = thread_position_in_threadgroup ; col < constants.dim ; col += threads_per_threadgroup ) {

        sum += ( A[ row * constants.dim + col ] * xin[ col ] );
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

            xout[row] = (b[row] - warp_sum)*Dinv[row];

            atomic_add_float( x_error, (xout[row] - xin[row])*(xout[row] - xin[row]) );
        }
    }
}








// - input vector cached
// - col major for coalesced load
// - one thread per row
// NOTE: Use of threadgroup memory in metal does not make it faster if the num rows exceeds 1K.

kernel void solve_col_major_old (

    device const float*              A                              [[ buffer(0) ]],

    device const float*              Dinv                           [[ buffer(1) ]],

    device const float*              b                              [[ buffer(2) ]],

    device const float*              xin                            [[ buffer(3) ]],

    device float*                    xout                           [[ buffer(4) ]],

    device float&                    x_error                        [[ buffer(5) ]],

    device const jacobi_solver_constants& constants                 [[ buffer(6) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]

) {
    const int THREADS_PER_THREADGROUP      = 1024;    // macos
    const int THREADGROUP_MEM_MAX_IN_FLOAT = 8* 1024; // macos

    // 1st step: xout = A*xi

    threadgroup float xin_cache[ THREADGROUP_MEM_MAX_IN_FLOAT ];

    for ( int col_base = 0; col_base < constants.dim; col_base += THREADGROUP_MEM_MAX_IN_FLOAT ) {

        for ( int row_base = 0; row_base < constants.dim; row_base += THREADS_PER_THREADGROUP ){

            for ( int i = 0; i < THREADGROUP_MEM_MAX_IN_FLOAT / THREADS_PER_THREADGROUP; i++ ) {

                const int col_base_xin_cache = i * THREADS_PER_THREADGROUP + thread_position_in_threadgroup;

                if ( col_base + col_base_xin_cache < constants.dim ) {

                    xin_cache[ col_base_xin_cache ] = xin[ col_base + col_base_xin_cache ];
                }
            }

            threadgroup_barrier( mem_flags::mem_threadgroup );

            const int row = row_base + thread_position_in_threadgroup;

            if ( row < constants.dim ) {

                const int col_end =   ( constants.dim - col_base < THREADGROUP_MEM_MAX_IN_FLOAT )
                                    ? ( constants.dim - col_base )
                                    : THREADGROUP_MEM_MAX_IN_FLOAT ;

                float sum = 0.0;

                for ( int k = 0; k < col_end; k++ ) {

                    sum += ( A[ row + constants.dim * ( col_base + k) ] * xin_cache[k] );
                }

                xout[row] += sum;
            }

            threadgroup_barrier( mem_flags::mem_device );
        }    
    }

    // 2nd step: xout = (b - xout) * Dinv
    for ( int i_base = 0; i_base < constants.dim; i_base += THREADS_PER_THREADGROUP ) {
        const int i = i_base + thread_position_in_threadgroup;
        if ( i < constants.dim ) {
            xout[i] = (b[i] - xout[i])*Dinv[i];
        }
    }
}


kernel void solve_row_major_old (

    device const float*              A                              [[ buffer(0) ]],

    device const float*              Dinv                           [[ buffer(1) ]],

    device const float*              b                              [[ buffer(2) ]],

    device const float*              xin                            [[ buffer(3) ]],

    device float*                    xout                           [[ buffer(4) ]],

    device float&                    x_error                        [[ buffer(5) ]],

    device const jacobi_solver_constants& constants                 [[ buffer(6) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint                threads_per_threadgroup        [[ threads_per_threadgroup ]]
) {

    const int THREADS_PER_THREADGROUP      = 1024;    // macos

    // 1st step: xout = A*xi

    threadgroup float sum_cache[ THREADS_PER_THREADGROUP ];

    for ( int row = 0; row < constants.dim; row++ ) {

        float sum = 0.0;
        for ( int col  = thread_position_in_threadgroup ; col < constants.dim ; col += threads_per_threadgroup ) {

            sum += ( A[ row * constants.dim + col ] * xin[ col ] );
        }
        sum_cache[ thread_position_in_threadgroup ] = sum;

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // reduce
        for ( int offset = threads_per_threadgroup / 2 ; offset >= 1; offset >>= 1 ) {

            if ( offset > 16 ) {

                if ( thread_position_in_threadgroup + offset < threads_per_threadgroup ) {

                    sum = sum_cache[thread_position_in_threadgroup] + sum_cache[thread_position_in_threadgroup + offset];
                    sum_cache[thread_position_in_threadgroup] = sum;

                }
                threadgroup_barrier( mem_flags::mem_threadgroup );

            }
            else {
                sum += simd_shuffle_down( sum, offset );
            }
        }

        if ( thread_position_in_threadgroup == 0 ) {

            xout[row] = sum;
        }
    }

    threadgroup_barrier( mem_flags::mem_device );

    // 2nd step: xout = (b - xout) * Dinv

    for ( int i_base = 0; i_base < constants.dim; i_base += THREADS_PER_THREADGROUP ) {

        const int i = i_base + thread_position_in_threadgroup;

        if ( i < constants.dim ) {

            xout[i] = (b[i] - xout[i])*Dinv[i];
        }
    }
}

#include <metal_stdlib>

using namespace metal;

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

struct gauss_seidel_solver_constants {
    int  dim;
};


kernel void solve_raw_major (

    device const float*              A                              [[ buffer(0) ]],

    device const float*              Dinv                           [[ buffer(1) ]],

    device const float*              b                              [[ buffer(2) ]],

    device const float*              xin                            [[ buffer(3) ]],

    device float*                    xout                           [[ buffer(4) ]],

    device atomic_uint*              x_error                        [[ buffer(5) ]],

    device const gauss_seidel_solver_constants& constants           [[ buffer(6) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]],

    const        uint                threads_per_threadgroup        [[ threads_per_threadgroup ]],

    const        uint                thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],

    const        uint                simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],

    const        uint                simdgroups_per_threadgroup     [[ simdgroups_per_threadgroup ]]
) {

    const int THREADS_PER_THREADGROUP      = 1024;    // macos

    // 1st step: xout = A*xi

    threadgroup float sum_cache[ THREADS_PER_THREADGROUP ];

    for ( int row = 0; row < constants.dim; row++ ) {

        float sum = 0.0;
        for ( int col  = thread_position_in_threadgroup ; col < constants.dim ; col += threads_per_threadgroup ) {
            if ( col < row ) {
                sum += ( A[ row * constants.dim + col ] * xout[ col ] );
            }
            else if ( col > row ) {
                sum += ( A[ row * constants.dim + col ] * xin[ col ] );
            }
        }
        const float warp_sum = simd_sum (sum);

        if ( thread_index_in_simdgroup == 0 ){

            sum_cache[ simdgroup_index_in_threadgroup ] = warp_sum;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        if ( simdgroup_index_in_threadgroup == 0 ) {

            const float local_sum =   (thread_index_in_simdgroup < simdgroups_per_threadgroup)
                                    ? sum_cache[ thread_index_in_simdgroup ]
                                    : 0.0;

            const float warp_sum =  simd_sum( local_sum );

            if ( thread_position_in_threadgroup == 0 ) {

                xout[row] = (b[row] - warp_sum)*Dinv[row];

                atomic_add_float( x_error, (xout[row] - xin[row])*(xout[row] - xin[row]) );
            }
        }
    }
}

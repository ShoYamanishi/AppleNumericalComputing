#include <metal_stdlib>

using namespace metal;


typedef struct conjugate_gradient_constants {
    int   dim;
    float epsilon;
    int   max_num_iterations;
} config;

kernel void conjugate_gradient (

    device float*        A                              [[ buffer(0) ]],
    device float*        x                              [[ buffer(1) ]],
    device float*        b                              [[ buffer(2) ]],
    device float*        r                              [[ buffer(3) ]],
    device float*        p                              [[ buffer(4) ]],
    device float*        Ap                             [[ buffer(5) ]],
    device const config& conf                           [[ buffer(6) ]],
    device int&          num_iterations                 [[ buffer(7) ]],

    const        uint    thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint    threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint    simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],
    const        uint    thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]]
) {

    threadgroup float rtr       = 0.0;
    threadgroup float pAp       = 0.0;
    threadgroup float alpha     = 0.0;
    threadgroup float beta      = 0.0;
    threadgroup bool  converged = false;

    threadgroup float scratch_array [32];
    threadgroup float scratch_array2[32];
    if ( thread_position_in_threadgroup  < 32 ) {

        scratch_array [thread_position_in_threadgroup] = 0.0;
        scratch_array2[thread_position_in_threadgroup] = 0.0;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // r := b - A * x
    // p := r

    for ( int row = thread_position_in_threadgroup; row < conf.dim; row += threads_per_threadgroup ) {

        float Ax = 0.0;
        for ( int col = 0; col < conf.dim; col++ ) {

            Ax += ( A[ row * conf.dim + col ] * x[ col ] );
        }

        r[ row ] = b[row] - Ax;
        p[ row ] = b[row] - Ax;
    }

    threadgroup_barrier( mem_flags::mem_device );

    // if r is sufficiently small, then return x as the result

    float r_max_local = 0.0;
    float rtr_local   = 0.0;

    for ( int row = thread_position_in_threadgroup; row < conf.dim; row += threads_per_threadgroup ) {

        r_max_local = max( r_max_local, fabs( r[row] ) );
        rtr_local += ( r[row] * r[row] );
    }

    float r_max_simdgroup = simd_max( r_max_local );
    float rtr_simdgroup   = simd_sum( rtr_local );

    if ( thread_index_in_simdgroup == 0 ){

        scratch_array [ simdgroup_index_in_threadgroup ] = r_max_simdgroup;
        scratch_array2[ simdgroup_index_in_threadgroup ] = rtr_simdgroup;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        r_max_simdgroup = scratch_array [ thread_index_in_simdgroup ];
        rtr_simdgroup   = scratch_array2[ thread_index_in_simdgroup ];

        thread const float r_max_threadgroup = simd_max( r_max_simdgroup );
        thread const float rtr_threadgroup   = simd_sum( rtr_simdgroup );

        if ( thread_position_in_threadgroup == 0 ) {

            converged = (r_max_threadgroup < conf.epsilon);
            rtr = rtr_threadgroup;
            num_iterations = 0;
        }
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( int i = 0; !converged && i < conf.max_num_iterations; i++ ) {

        // Ap := A * p

        float pAp_local = 0.0;

        for ( int row = thread_position_in_threadgroup; row < conf.dim; row += threads_per_threadgroup ) {

            float Ap_local = 0.0;

            for ( int col = 0; col < conf.dim; col++ ) {

                Ap_local += ( A[ row * conf.dim + col ] * p[ col ] );
            }
            Ap[ row ] = Ap_local;

            pAp_local += ( p[ row ] * Ap_local );
        }

        threadgroup_barrier( mem_flags::mem_device );


        // alpha := rtr / pAp

        float pAp_simdgroup = simd_sum( pAp_local );

        if ( thread_index_in_simdgroup == 0 ){

            scratch_array[ simdgroup_index_in_threadgroup ] = pAp_simdgroup;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        if ( simdgroup_index_in_threadgroup == 0 ) {

            thread const float pAp_simdgroup   = scratch_array[ thread_index_in_simdgroup ];
            thread const float pAp_threadgroup = simd_sum( pAp_simdgroup );

            if ( thread_position_in_threadgroup == 0 ) {

                pAp   = pAp_threadgroup;
                alpha = rtr / pAp_threadgroup;
            }
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        float r_new_max_local = 0.0;
        float rtr_new_local   = 0.0;

        // x := x + alpha * p
        // r := r - alpha * Ap

        for ( int row = thread_position_in_threadgroup; row < conf.dim; row += threads_per_threadgroup ) {

            x[ row ] = x[ row ] + alpha * p[ row ];

            const float r_new = r[ row ] - alpha * Ap[ row ];

            r[ row ] = r_new;
            
            r_new_max_local = max( r_new_max_local, fabs( r_new ) );
            rtr_new_local += ( r_new * r_new );
        }

        threadgroup_barrier( mem_flags::mem_device );

        // beta := rtr_new / rtr

        float r_new_max_simdgroup = simd_max( r_new_max_local );
        float rtr_new_simdgroup   = simd_sum( rtr_new_local );

        if ( thread_index_in_simdgroup == 0 ){

            scratch_array [ simdgroup_index_in_threadgroup ] = r_new_max_simdgroup;
            scratch_array2[ simdgroup_index_in_threadgroup ] = rtr_new_simdgroup;
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        if ( simdgroup_index_in_threadgroup == 0 ) {

            thread const float r_new_max_simdgroup = scratch_array [ thread_index_in_simdgroup ];
            thread const float rtr_new_simdgroup   = scratch_array2[ thread_index_in_simdgroup ];

            thread const float r_new_max_threadgroup = simd_max( r_new_max_simdgroup );
            thread const float rtr_new_threadgroup   = simd_sum( rtr_new_simdgroup );

            if ( thread_position_in_threadgroup == 0 ) {

                converged      = (r_new_max_threadgroup < conf.epsilon);
                beta           = rtr_new_threadgroup / rtr;
                rtr            = rtr_new_threadgroup;
                num_iterations = i + 1;
            }
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // p := r + beta * p

        for ( int row = thread_position_in_threadgroup; row < conf.dim; row += threads_per_threadgroup ) {

            p[ row ] = r[ row ] + beta * p[ row ];
        }

        threadgroup_barrier( mem_flags::mem_device );
    }
}


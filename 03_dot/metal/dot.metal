#include <metal_stdlib>

using namespace metal;

struct dot_constants
{
    uint  num_elements;
};

/**
 *  Type 1, 2 & 3 : two-pass
 *
 *  Let N_gpg be the number of groups per grid, N_tpg be the number of threads per group,
 *  and N_elem be the number of elements in X and Y.   
 * 
 * In the first pass, each thread (in a group) sums up X[i] * Y[i],
 * where i runs from therad_position_in_grid, upwards with increment of N_gpg * N_tpg, 
 * i.e., the number of threads per grid.
 * At the end of the first pass, Z contains the partial sumbs calculated above.
 * The size of Z is Ngpg, and each element represents the partial sum of therad_position_in_grid.
 *
 * The second pass launches only one thread group, i.e., number of thread groups per grid = 1.
 * It reduces Z down to one total sum.
 *
 * value of dot_constants::num_elements          : N_elem
 * Size of X & Y                                 : N_elem
 * Size of s_partials [[ threadgroup(0) ]]       : N_tpg
 * Size of Z                                     : N_gpg
 * The configuration of the first kernel launch  : <<Ngpg, Ntpg >>
 * The configuration of the second kernel launch : <<1, Ntpg >>
 */

kernel void dot_type1_pass1(

    device const float*           X          [[ buffer(0) ]],
    device const float*           Y          [[ buffer(1) ]],
    device       float*           Z          [[ buffer(2) ]],
    device const dot_constants&   c          [[ buffer(3) ]],
    device       float*           s_partials [[ buffer(4) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]]
) {
    float sum = 0;

    for ( size_t i =  thread_position_in_grid;
                 i <  c.num_elements;
                 i += threads_per_grid
    ) {
        sum += (X[i]*Y[i]);
    }

//    s_partials[ thread_position_in_threadgroup ] = sum;
    s_partials[ thread_position_in_grid ] = sum;

    threadgroup_barrier( mem_flags::mem_device );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 1;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_grid ] += 
                s_partials[ thread_position_in_grid + activeThreads ];
        }

        threadgroup_barrier( mem_flags::mem_device );
    }

    if ( thread_position_in_threadgroup == 0 ) {

        Z[ threadgroup_position_in_grid ] = s_partials[ thread_position_in_grid ];
    }
}


kernel void dot_type1_pass2(

    device const float*           Z          [[ buffer(0) ]],
    device       float*           dot        [[ buffer(1) ]],
    device const dot_constants&   c          [[ buffer(2) ]],
    device       float*           s_partials [[ buffer(3) ]],

    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]]

) {
    float sum = 0.0;

    for ( size_t i = thread_position_in_threadgroup;
                 i < c.num_elements;
                 i+= threads_per_threadgroup
    ) {
        sum += Z[i];
    }

    s_partials[ thread_position_in_threadgroup ] = sum;

    threadgroup_barrier( mem_flags::mem_device );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 1;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }
        threadgroup_barrier( mem_flags::mem_device );
    }

    if ( thread_position_in_threadgroup == 0 ) {

        dot[0] = s_partials[0];        
    }
}


kernel void dot_type2_threadgroup_memory_pass1(

    device const float*           X          [[ buffer(0) ]],
    device const float*           Y          [[ buffer(1) ]],
    device       float*           Z          [[ buffer(2) ]],
    device const dot_constants&   c          [[ buffer(3) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]]
) {
    float sum = 0;

    for ( size_t i =  thread_position_in_grid;
                 i <  c.num_elements;
                 i += threads_per_grid
    ) {
        sum += (X[i]*Y[i]);
    }

    s_partials[ thread_position_in_threadgroup ] = sum;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 1;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }
        if ( activeThreads >= threads_per_simdgroup ) {

            threadgroup_barrier( mem_flags::mem_threadgroup );
        }
        else{
            simdgroup_barrier( mem_flags::mem_threadgroup );
        }
    }

    if ( thread_position_in_threadgroup == 0 ) {

        Z[ threadgroup_position_in_grid ] = s_partials[0];
    }
}


kernel void dot_type2_threadgroup_memory_pass2(

    device const float*           Z          [[ buffer(0) ]],
    device       float*           dot        [[ buffer(1) ]],
    device const dot_constants&   c          [[ buffer(2) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]]

) {
    float sum = 0.0;

    for ( size_t i = thread_position_in_threadgroup;
                 i < c.num_elements;
                 i+= threads_per_threadgroup
    ) {
        sum += Z[i];
    }

    s_partials[ thread_position_in_threadgroup ] = sum;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 1;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }
        if ( activeThreads >= threads_per_simdgroup ){

            threadgroup_barrier( mem_flags::mem_threadgroup );
        }
        else{
            simdgroup_barrier( mem_flags::mem_threadgroup );
        }
    }

    if ( thread_position_in_threadgroup == 0 ) {

        dot[0] = s_partials[0];        
    }
}


kernel void dot_type3_pass1_simd_shuffle(

    device const float*           X          [[ buffer(0) ]],
    device const float*           Y          [[ buffer(1) ]],
    device       float*           Z          [[ buffer(2) ]],
    device const dot_constants&   c          [[ buffer(3) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]]
) {

    float sum = 0;

    for ( size_t i =  thread_position_in_grid;
                 i <  c.num_elements;
                 i += threads_per_grid
    ) {

        sum += (X[i]*Y[i]);
    }

    s_partials[ thread_position_in_threadgroup ] = sum;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 32;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }
        threadgroup_barrier( mem_flags::mem_threadgroup );
    }

    float simd_sum = s_partials[ thread_position_in_threadgroup ];

    simd_sum += simd_shuffle_xor( simd_sum, 16 );
    simd_sum += simd_shuffle_xor( simd_sum,  8 );
    simd_sum += simd_shuffle_xor( simd_sum,  4 );
    simd_sum += simd_shuffle_xor( simd_sum,  2 );
    simd_sum += simd_shuffle_xor( simd_sum,  1 );

    if ( thread_position_in_threadgroup == 0 ) {

        Z[ threadgroup_position_in_grid ] = simd_sum;
    }
}


kernel void dot_type3_pass2_simd_shuffle(

    device const float*           Z          [[ buffer(0) ]],
    device       float*           dot        [[ buffer(1) ]],
    device const dot_constants&   c          [[ buffer(2) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]]
) {
    float sum = 0.0;

    for ( size_t i = thread_position_in_threadgroup;
                 i < c.num_elements;
                 i+= threads_per_threadgroup
    ) {
        sum += Z[i];
    }

    s_partials[thread_position_in_threadgroup] = sum;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 32;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );
    }

    float simd_sum = s_partials[ thread_position_in_threadgroup ];

    simd_sum += simd_shuffle_xor( simd_sum, 16 );
    simd_sum += simd_shuffle_xor( simd_sum,  8 );
    simd_sum += simd_shuffle_xor( simd_sum,  4 );
    simd_sum += simd_shuffle_xor( simd_sum,  2 );
    simd_sum += simd_shuffle_xor( simd_sum,  1 );

    if ( thread_position_in_threadgroup == 0 ) {

        dot[0] = simd_sum;
    }
}


kernel void dot_type4_pass1_simd_add(

    device const float*           X          [[ buffer(0) ]],
    device const float*           Y          [[ buffer(1) ]],
    device       float*           Z          [[ buffer(2) ]],
    device const dot_constants&   c          [[ buffer(3) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]],
    const        uint             thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],
    const        uint             simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]]
) {

    float sum = 0;

    for ( size_t i = thread_position_in_grid; i < c.num_elements; i += threads_per_grid ) {

        sum += (X[i]*Y[i]);
    }

    // Reset all the 32 elements in case threads_per_threadgroup < 1024.
    if ( simdgroup_index_in_threadgroup == 0 ) { 
        s_partials[thread_index_in_simdgroup] = 0.0;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    thread const float local_sum = simd_sum(sum);

    if ( thread_index_in_simdgroup == 0 ){
        s_partials[simdgroup_index_in_threadgroup] = local_sum;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        thread const float local_sum2 = s_partials[ thread_index_in_simdgroup ];
        thread const float warp_sum   = simd_sum(local_sum2);

        if ( thread_position_in_threadgroup == 0 ) {
            Z[ threadgroup_position_in_grid ] = warp_sum;
        }
    }
}


kernel void dot_type4_pass2_simd_add(

    device const float*           Z          [[ buffer(0) ]],
    device       float*           dot        [[ buffer(1) ]],
    device const dot_constants&   c          [[ buffer(2) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]],
    const        uint             thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]],
    const        uint             simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]]
) {
    float sum = 0.0;

    for ( size_t i = thread_position_in_threadgroup; i < c.num_elements; i += threads_per_threadgroup ) {

        sum += Z[i];
    }

    // Reset all the 32 elements in case threads_per_threadgroup < 1024.
    if ( simdgroup_index_in_threadgroup == 0 ) { 
        s_partials[thread_index_in_simdgroup] = 0.0;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    thread const float local_sum = simd_sum(sum);

    if ( thread_index_in_simdgroup == 0 ){
        s_partials[simdgroup_index_in_threadgroup] = local_sum;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {

        thread const float local_sum2 = s_partials[ thread_index_in_simdgroup ];
        thread const float warp_sum   = simd_sum(local_sum2);

        if ( thread_position_in_threadgroup == 0 ) {
            dot[ 0 ] = warp_sum;
        }
    }
}


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


kernel void dot_type5_atomic_simd_shuffle(

    device const    float*          X        [[ buffer(0) ]],
    device const    float*          Y        [[ buffer(1) ]],
    device          atomic_uint*    Sint     [[ buffer(2) ]],
    device const    dot_constants&  c        [[ buffer(3) ]],

    threadgroup  float*           s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]],
    const        uint             threadgroups_per_grid          [[ threadgroups_per_grid ]],
    const        uint             simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],
    const        uint             thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]]
) {
    float sum = 0;

    for ( size_t i =  thread_position_in_grid;
                 i <  c.num_elements;
                 i += threads_per_grid
    ) {
        sum += (X[i]*Y[i]);
    }

    s_partials[ thread_position_in_threadgroup ] = sum;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 32;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }
        threadgroup_barrier( mem_flags::mem_threadgroup );
    }

    float simd_sum = s_partials[ thread_position_in_threadgroup ];

    simd_sum += simd_shuffle_xor( simd_sum, 16 );
    simd_sum += simd_shuffle_xor( simd_sum,  8 );
    simd_sum += simd_shuffle_xor( simd_sum,  4 );
    simd_sum += simd_shuffle_xor( simd_sum,  2 );
    simd_sum += simd_shuffle_xor( simd_sum,  1 );

    if ( thread_position_in_threadgroup == 0 ) {

        atomic_add_float( Sint, simd_sum );
    }
}


kernel void dot_type6_atomic_simd_add(

    device const    float*          X          [[ buffer(0) ]],
    device const    float*          Y          [[ buffer(1) ]],
    device          atomic_uint*    Sint       [[ buffer(2) ]],
    device const    dot_constants&  c          [[ buffer(3) ]],

    threadgroup  float*             s_partials [[ threadgroup(0) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]],
    const        uint             threadgroups_per_grid          [[ threadgroups_per_grid ]],
    const        uint             simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]],
    const        uint             thread_index_in_simdgroup      [[ thread_index_in_simdgroup ]]
) {
    float sum = 0;

    for ( size_t i =  thread_position_in_grid;
                 i <  c.num_elements;
                 i += threads_per_grid
    ) {
        sum += (X[i]*Y[i]);
    }

    // Reset all the 32 elements in case threads_per_threadgroup < 1024.
    if ( simdgroup_index_in_threadgroup == 0 ) { 
        s_partials[thread_index_in_simdgroup] = 0.0;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    thread const float local_sum = simd_sum(sum);
    thread const float warp_sum  = local_sum;

    if ( thread_index_in_simdgroup == 0 ){
        s_partials[simdgroup_index_in_threadgroup] = warp_sum;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simdgroup_index_in_threadgroup == 0 ) {
        thread const float local_sum2 = s_partials[ thread_index_in_simdgroup ];
        thread const float warp_sum2  = simd_sum(local_sum2);

        if ( thread_position_in_threadgroup == 0 ) {

            atomic_add_float( Sint, warp_sum2 );
        }
    }
}


/** Type 6 One-pass
 *  This is based on 12.3 Single-Pass-Reduction of "The CUDA Handbook" by Wilt.
 *
 * The reason is why it does not work is that Z buffer is not synchronized between Point A and B below.
 * After writing to Z[i], we call threadgroup_barrier( mem_flags::mem_device ) at Point A,
 * However the memory consistency is apparently kept within the thread group only.
 * Even if all the other thread grouops finished execution, i.e., finished writing to Z and calling
 * threadgroup_barrier( mem_flags::mem_device ) at Point A, 
 * the values the other threadgroup than the last one have written to Z are not visible to the last thread group.
 * 
 * This behavior is different from CUDA where at point A _threadFence() would be called to guarantee Grid-wise memory consistency.
 *
 * It seems you have to split it into two serial kernel invocations along the same compute buffer
 * to ensure the grid-wise consistency, which will make it essencially Type 1 or 2.
 */
kernel void dot_type7_atomic_thread_group_counter(

    device const    float*        X          [[ buffer(0) ]],
    device const    float*        Y          [[ buffer(1) ]],
    device volatile float*        Z          [[ buffer(2) ]],
    device       float*           dot        [[ buffer(3) ]],
    device       atomic_uint*     thread_group_counter 
                                             [[ buffer(4) ]],
    device const dot_constants&   c          [[ buffer(5) ]],

    const        uint             thread_position_in_grid        [[ thread_position_in_grid ]],
    const        uint             thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    const        uint             threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    const        uint             threads_per_threadgroup        [[ threads_per_threadgroup ]],
    const        uint             threads_per_simdgroup          [[ threads_per_simdgroup ]],
    const        uint             threads_per_grid               [[ threads_per_grid ]],
    const        uint             threadgroups_per_grid          [[ threadgroups_per_grid ]]
) {

    threadgroup int   last_group = 0; // 'bool' does not work for threadgroup somehow. Use 'int' instead.
    threadgroup float s_partials [1024];


    float sum = 0;

    for ( size_t i =  thread_position_in_grid;
                 i <  c.num_elements;
                 i += threads_per_grid
    ) {
        sum += (X[i]*Y[i]);
    }

    s_partials[ thread_position_in_threadgroup ] = sum;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 32;
               activeThreads >>= 1
    ) {
        if ( thread_position_in_threadgroup < activeThreads ) {

            s_partials[ thread_position_in_threadgroup ] += 
                s_partials[ thread_position_in_threadgroup + activeThreads ];
        }
        threadgroup_barrier( mem_flags::mem_threadgroup );
    }

    float simd_sum = s_partials[ thread_position_in_threadgroup ];

    simd_sum += simd_shuffle_xor( simd_sum, 16 );
    simd_sum += simd_shuffle_xor( simd_sum,  8 );
    simd_sum += simd_shuffle_xor( simd_sum,  4 );
    simd_sum += simd_shuffle_xor( simd_sum,  2 );
    simd_sum += simd_shuffle_xor( simd_sum,  1 );

    if ( thread_position_in_threadgroup == 0 ) {

        Z[ threadgroup_position_in_grid ] = simd_sum;
    }

    threadgroup_barrier( mem_flags::mem_device ); // <= (Point A)

    uint fetched_val = 0;

    if ( thread_position_in_threadgroup == 0 ) {

        fetched_val = atomic_fetch_add_explicit( thread_group_counter, 1 , memory_order_relaxed );

        last_group = ( fetched_val == threadgroups_per_grid - 1 )?1:0 ;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( last_group == 1 ) {

        sum = 0.0;

        for ( size_t i = thread_position_in_threadgroup;
                     i < threadgroups_per_grid;
                     i+= threads_per_threadgroup
        ) {
            sum += Z[i];  // <= (Point B)
        }

        s_partials[ thread_position_in_threadgroup ] = sum;

        threadgroup_barrier( mem_flags::mem_threadgroup );

        for ( uint activeThreads = threads_per_threadgroup >> 1;
               activeThreads >= 32;
               activeThreads >>= 1
        ) {
            if ( thread_position_in_threadgroup < activeThreads ) {

                s_partials[ thread_position_in_threadgroup ] += 
                    s_partials[ thread_position_in_threadgroup + activeThreads ];
            }
            threadgroup_barrier( mem_flags::mem_threadgroup );
        }

        float simd_sum = s_partials[ thread_position_in_threadgroup ];

        simd_sum += simd_shuffle_xor( simd_sum, 16 );
        simd_sum += simd_shuffle_xor( simd_sum,  8 );
        simd_sum += simd_shuffle_xor( simd_sum,  4 );
        simd_sum += simd_shuffle_xor( simd_sum,  2 );
        simd_sum += simd_shuffle_xor( simd_sum,  1 );

        if ( thread_position_in_threadgroup == 0 ) {

            dot[0] = simd_sum;
        }
    }
}

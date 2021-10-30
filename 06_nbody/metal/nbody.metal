#include <metal_stdlib>

using namespace metal;

struct nbody_constants {
    uint  num_elements;
    float EPSILON;
    float COEFF_G;
    float delta_t;
};

struct particle {

    float3 p0; // point 1 current
    float3 p1; // point 2 new
    float3 v;  // velocity
    float4 am; // acceleration(xyz) and mass(w)
};

struct particle_constants {

    float4 pm; // point(xyz) and mass(w)
    float3 v;  // velocity
};

struct particle_updates {

    float3 p;  // point
    float3 v;  // velocity
    float3 a;  // acceleration
};


inline void body_body_interaction_p0_to_p1( thread float3& a, device struct particle& pi, device const struct particle& pj, device const struct nbody_constants& c )
{
    const float3 d     = pj.p0 - pi.p0;

//    const float  d_len = length( d ) + c.EPSILON;
//    const float  s = pj.am.w / ( d_len * d_len * d_len );

    const float d_len_sq  = d.x * d.x + d.y * d.y + d.z * d.z + 1.0e-5;
    const float d_len_inv = rsqrt(d_len_sq);
    const float d_len_inv_cube = d_len_inv *  d_len_inv *  d_len_inv ;
    const float  s = pj.am.w * d_len_inv_cube;

    a.x = d.x * s;
    a.y = d.y * s;
    a.z = d.z * s;
}


inline void body_body_interaction_p1_to_p0( thread float3& a, device struct particle& pi, device const struct particle& pj, device const struct nbody_constants& c )
{
    const float3 d     = pj.p1 - pi.p1;

//    const float  d_len = length( d ) + c.EPSILON;
//    const float  s = pj.am.w / ( d_len * d_len * d_len );

    const float d_len_sq  = d.x * d.x + d.y * d.y + d.z * d.z + 1.0e-5;
    const float d_len_inv = rsqrt(d_len_sq);
    const float d_len_inv_cube = d_len_inv *  d_len_inv *  d_len_inv ;
    const float  s = pj.am.w * d_len_inv_cube;

    a.x = d.x * s;
    a.y = d.y * s;
    a.z = d.z * s;
}


kernel void nbody_naive_p0_to_p1(

    device struct particle*       particles                     [[ buffer(0) ]],

    device const nbody_constants& constants                     [[ buffer(1) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]]
) {
    if ( thread_position_in_grid < constants.num_elements ) {

        const uint i = thread_position_in_grid;

        device auto& pi = particles[i];

        thread float3 a_sum{0.0, 0.0, 0.0};

        // NOTE: manual unrolling beyond order 4 gives runtime error somehow..
        //       and manual unrolling does not show any performance advantage.
        //       not worth it.
        for ( uint j = 0; j < constants.num_elements ; j++) {

            device const auto& pj1 = particles[j];

            thread float3 a1;

            body_body_interaction_p0_to_p1( a1, pi, pj1, constants );

            if (i!=j) {// Make the branching as	small as possible so that it can use conditional store.
                a_sum += a1;
            }
        }

        pi.am.x = a_sum.x;
        pi.am.y = a_sum.y;
        pi.am.z = a_sum.z;

        pi.v.x += ( a_sum.x * pi.am.w * constants.COEFF_G * constants.delta_t );
        pi.v.y += ( a_sum.y * pi.am.w * constants.COEFF_G * constants.delta_t );
        pi.v.z += ( a_sum.z * pi.am.w * constants.COEFF_G * constants.delta_t );

        pi.p1.x = pi.p0.x + pi.v.x * constants.delta_t;
        pi.p1.y = pi.p0.y + pi.v.y * constants.delta_t;
        pi.p1.z = pi.p0.z + pi.v.z * constants.delta_t;
    }
}


kernel void nbody_naive_p1_to_p0(

    device struct particle*       particles                     [[ buffer(0) ]],

    device const nbody_constants& constants                     [[ buffer(1) ]],

    const        uint            thread_position_in_grid        [[ thread_position_in_grid ]]
) {
    if ( thread_position_in_grid < constants.num_elements ) {

        const uint i = thread_position_in_grid;

        device auto& pi = particles[i];

        thread float3 a_sum{0.0, 0.0, 0.0};

        for ( uint j = 0; j < constants.num_elements ; j++ ) {

            device const auto& pj = particles[j];

            thread float3 a;

            body_body_interaction_p1_to_p0( a, pi, pj, constants );

            if (i!=j) {// Make the branching as	small as possible so that it can use conditional store.
                a_sum += a;
            }
        }

        pi.am.x = a_sum.x;
        pi.am.y = a_sum.y;
        pi.am.z = a_sum.z;

        pi.v.x += ( a_sum.x * pi.am.w * constants.COEFF_G * constants.delta_t );
        pi.v.y += ( a_sum.y * pi.am.w * constants.COEFF_G * constants.delta_t );
        pi.v.z += ( a_sum.z * pi.am.w * constants.COEFF_G * constants.delta_t );

        pi.p0.x = pi.p1.x + pi.v.x * constants.delta_t;
        pi.p0.y = pi.p1.y + pi.v.y * constants.delta_t;
        pi.p0.z = pi.p1.z + pi.v.z * constants.delta_t;
    }
}

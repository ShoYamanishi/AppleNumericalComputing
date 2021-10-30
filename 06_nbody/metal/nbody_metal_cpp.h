#ifndef __NBODY_METAL_CPP_H__
#define __NBODY_METAL_CPP_H__

#include <simd/simd.h>
typedef unsigned int uint;
typedef simd_float3 float3;
typedef simd_float4 float4;


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

class NBodyMetalCppImpl;

class NBodyMetalCpp
{

  public:
    NBodyMetalCpp( const size_t num_elements );

    virtual ~NBodyMetalCpp();

    unsigned int numElements();

    struct particle* getRawPointerParticles();

    void performComputationDirectionIsP0ToP1( const bool p0_to_p1 );

  private:
    NBodyMetalCppImpl* m_impl;

};

#endif /*__NBODY_METAL_CPP_H__*/

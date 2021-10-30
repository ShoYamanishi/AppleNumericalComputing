#ifndef __NBODY_METAL_CPP_IMPL_H__
#define __NBODY_METAL_CPP_IMPL_H__

#include <nbody_metal_cpp.h>

#include <cstddef>

class NBodyMetalCppImpl
{

  public:
    NBodyMetalCppImpl( const size_t num_elements );

    virtual ~NBodyMetalCppImpl();

    unsigned int numElements();

    struct particle* getRawPointerParticles();

    void performComputationDirectionIsP0ToP1( const bool p0_to_p1 );

  private:
    void * m_self;

};

#endif /*__NBODY_METAL_CPP_IMPL_H__*/

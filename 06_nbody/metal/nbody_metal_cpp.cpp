#include "nbody_metal_cpp_impl.h"
#include "nbody_metal_cpp.h"

NBodyMetalCpp::NBodyMetalCpp( const size_t num_elements )
    :m_impl( new NBodyMetalCppImpl( num_elements ) )
{
    ;
}

NBodyMetalCpp::~NBodyMetalCpp()
{
    delete m_impl;
};

unsigned int NBodyMetalCpp::numElements(){ return m_impl->numElements(); }

struct particle* NBodyMetalCpp::getRawPointerParticles()
{
    return m_impl->getRawPointerParticles();
}

void NBodyMetalCpp::performComputationDirectionIsP0ToP1( const bool p0_to_p1 )
{
    m_impl->performComputationDirectionIsP0ToP1( p0_to_p1 );
}

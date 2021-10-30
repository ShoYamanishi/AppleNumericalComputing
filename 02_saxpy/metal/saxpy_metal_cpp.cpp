#include "saxpy_metal_cpp_impl.h"
#include "saxpy_metal_cpp.h"

SaxpyMetalCpp::SaxpyMetalCpp( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid )
    :m_impl( new SaxpyMetalCppImpl( num_elements, num_threads_per_group, num_groups_per_grid ) )
{
    ;
}

SaxpyMetalCpp::~SaxpyMetalCpp()
{
    delete m_impl;
};

float* SaxpyMetalCpp::getRawPointerX()
{
    return m_impl->getRawPointerX();
}

float* SaxpyMetalCpp::getRawPointerY()
{
    return m_impl->getRawPointerY();
}

void SaxpyMetalCpp::setScalar_a( const float a )
{
    m_impl->setScalar_a( a );
}

void SaxpyMetalCpp::performComputation()
{
    m_impl->performComputation();
}

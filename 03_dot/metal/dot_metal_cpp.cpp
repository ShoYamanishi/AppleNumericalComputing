class DotMetalObjC;
#include "dot_metal_cpp_impl.h"
#include "dot_metal_cpp.h"

DotMetalCpp::DotMetalCpp( const size_t num_elements, const size_t num_threads_per_group, const size_t num_groups_per_grid, const int reduction_type )
    :m_impl( new DotMetalCppImpl( num_elements, num_threads_per_group, num_groups_per_grid, reduction_type ) )
{
    ;
}

DotMetalCpp::~DotMetalCpp()
{
    delete m_impl;
};

float* DotMetalCpp::getRawPointerX()
{
    return m_impl->getRawPointerX();
}

float* DotMetalCpp::getRawPointerY()
{
    return m_impl->getRawPointerY();
}

float DotMetalCpp::getDotXY()
{
    return m_impl->getDotXY();
}

void DotMetalCpp::performComputation()
{
    m_impl->performComputation();
}

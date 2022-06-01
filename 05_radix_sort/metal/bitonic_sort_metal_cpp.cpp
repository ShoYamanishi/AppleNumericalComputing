
class BitonicSortMetalObjC;
#include "bitonic_sort_metal_cpp_impl.h"
#include "bitonic_sort_metal_cpp.h"

BitonicSortMetalCpp::BitonicSortMetalCpp(
    const size_t num_elements,
    const bool   for_float, 
    const size_t num_threads_per_threadgroup
)
    :m_impl( new BitonicSortMetalCppImpl( num_elements , for_float , num_threads_per_threadgroup ) )
{
    ;
}

BitonicSortMetalCpp::~BitonicSortMetalCpp()
{
    delete m_impl;
};

unsigned int BitonicSortMetalCpp::numElements()
{
    return m_impl->numElements();
}

int* BitonicSortMetalCpp::getRawPointerInOut()
{
    return m_impl->getRawPointerInOut();
}

void BitonicSortMetalCpp::performComputation()
{
    m_impl->performComputation();
}

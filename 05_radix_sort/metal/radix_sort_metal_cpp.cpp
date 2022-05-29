class RadixSortMetalObjC;
#include "radix_sort_metal_cpp_impl.h"
#include "radix_sort_metal_cpp.h"

RadixSortMetalCpp::RadixSortMetalCpp(
    const size_t num_elements,
    const bool   for_float, 
    const bool   coalesced_write, 
    const bool   early_out, 
    const int    num_iterations_per_commit,
    const size_t num_threads_per_threadgroup
)
    :m_impl( new RadixSortMetalCppImpl( num_elements , for_float , coalesced_write, early_out, num_iterations_per_commit, num_threads_per_threadgroup ) )
{
    ;
}

RadixSortMetalCpp::~RadixSortMetalCpp()
{
    delete m_impl;
};

void RadixSortMetalCpp::resetBufferFlag()
{
    return m_impl->resetBufferFlag();
}

unsigned int RadixSortMetalCpp::numElements()
{
    return m_impl->numElements();
}

int* RadixSortMetalCpp::getRawPointerIn()
{
    return m_impl->getRawPointerIn();
}

int* RadixSortMetalCpp::getRawPointerOut()
{
    return m_impl->getRawPointerOut();
}

int* RadixSortMetalCpp::getRawPointerIn1()
{
    return m_impl->getRawPointerIn1();
}

int* RadixSortMetalCpp::getRawPointerIn2()
{
    return m_impl->getRawPointerIn2();
}

void RadixSortMetalCpp::performComputation()
{
    m_impl->performComputation();
}

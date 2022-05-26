class PrefixSumMetalObjCRecursive;
class PrefixSumMetalObjCMerrillGrimshaw;

#include "prefix_sum_metal_cpp_impl.h"
#include "prefix_sum_metal_cpp.h"

template<>
PrefixSumMetalCpp<int>::PrefixSumMetalCpp(
    const size_t num_elements,
    const int    algo_type,
    const size_t num_partial_sums,
    const int    num_threads_per_threadgroup
)
    :m_impl( new PrefixSumMetalCppImpl<int>( num_elements, algo_type, num_partial_sums, num_threads_per_threadgroup ) )
{
    ;
}

template<>
PrefixSumMetalCpp<float>::PrefixSumMetalCpp(
    const size_t num_elements,
    const int    algo_type,
    const size_t num_partial_sums,
    const int    num_threads_per_threadgroup
)
    :m_impl( new PrefixSumMetalCppImpl<float>( num_elements, algo_type, num_partial_sums, num_threads_per_threadgroup ) )
{
    ;
}

template<>
PrefixSumMetalCpp<int>::~PrefixSumMetalCpp()
{
    delete m_impl;
};

template<>
PrefixSumMetalCpp<float>::~PrefixSumMetalCpp()
{
    delete m_impl;
};

template<>
unsigned int PrefixSumMetalCpp<int>::numElements( unsigned int layer ){ return m_impl->numElements(layer); }

template<>
unsigned int PrefixSumMetalCpp<float>::numElements( unsigned int layer ){ return m_impl->numElements(layer); }

template<>
unsigned int PrefixSumMetalCpp<int>::numThreadsPerGroup( unsigned int layer ){ return m_impl->numThreadsPerGroup(layer); }

template<>
unsigned int PrefixSumMetalCpp<float>::numThreadsPerGroup( unsigned int layer ){ return m_impl->numThreadsPerGroup(layer); }

template<>
unsigned int PrefixSumMetalCpp<int>::numGroupsPerGrid( unsigned int layer ){ return m_impl->numGroupsPerGrid(layer); }

template<>
unsigned int PrefixSumMetalCpp<float>::numGroupsPerGrid( unsigned int layer ){ return m_impl->numGroupsPerGrid(layer); }

template<>
int* PrefixSumMetalCpp<int>::getRawPointerIn()
{
    return m_impl->getRawPointerIn();
}

template<>
float* PrefixSumMetalCpp<float>::getRawPointerIn()
{
    return m_impl->getRawPointerIn();
}

template<>
int* PrefixSumMetalCpp<int>::getRawPointerOut()
{
    return m_impl->getRawPointerOut();

}

template<>
float* PrefixSumMetalCpp<float>::getRawPointerOut()
{
    return m_impl->getRawPointerOut();

}

template<>
int* PrefixSumMetalCpp<int>::getRawPointerGridPrefixSums(unsigned int layer)
{
    return m_impl->getRawPointerGridPrefixSums(layer);
}

template<>
float* PrefixSumMetalCpp<float>::getRawPointerGridPrefixSums(unsigned int layer)
{
    return m_impl->getRawPointerGridPrefixSums(layer);
}

template<>
int* PrefixSumMetalCpp<int>::getRawPointerPartialSumsMerrillGrimshaw()
{
    return m_impl->getRawPointerPartialSumsMerrillGrimshaw();
}

template<>
float* PrefixSumMetalCpp<float>::getRawPointerPartialSumsMerrillGrimshaw()
{
    return m_impl->getRawPointerPartialSumsMerrillGrimshaw();
}

template<>
void PrefixSumMetalCpp<int>::performComputation()
{
    m_impl->performComputation();
}

template<>
void PrefixSumMetalCpp<float>::performComputation()
{
    m_impl->performComputation();
}

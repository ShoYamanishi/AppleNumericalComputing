#import "prefix_sum_metal_objc_recursive.h"
#import "prefix_sum_metal_objc_merrill_grimshaw.h"
#import "prefix_sum_metal_cpp_impl.h"

template<>
PrefixSumMetalCppImpl<int>::PrefixSumMetalCppImpl(
    const size_t num_elements,
    const int    algo_type,
    const size_t num_partial_sums,
    const int    num_threads_per_threadgroup 
) {
    m_algo_type = algo_type;

    if ( algo_type == 3 ) {

        m_self_merrill_grimshaw
              = [ [ PrefixSumMetalObjCMerrillGrimshaw alloc ] initWithNumElements : num_elements
                                                                   NumPartialSums : num_partial_sums 
                                                                          ForFloat: false
                                                          NumThreadsPerThreadgroup: num_threads_per_threadgroup ];
    }
    else {  
        m_self_recursive
              = [ [ PrefixSumMetalObjCRecursive alloc ] initWithNumElements : num_elements
                                                                       Type : algo_type
                                                             NumPartialSums : num_partial_sums 
                                                                   ForFloat : false
                                                    NumThreadsPerThreadgroup: num_threads_per_threadgroup ];
    }
}


template<>
PrefixSumMetalCppImpl<float>::PrefixSumMetalCppImpl(
    const size_t num_elements,
    const int    algo_type,
    const size_t num_partial_sums,
    const int    num_threads_per_threadgroup 
) {
    m_algo_type = algo_type;

    if ( algo_type == 3 ) {

        m_self_merrill_grimshaw
              = [ [ PrefixSumMetalObjCMerrillGrimshaw alloc ] initWithNumElements : num_elements
                                                                   NumPartialSums : num_partial_sums
                                                                          ForFloat: true
                                                          NumThreadsPerThreadgroup: num_threads_per_threadgroup ];
    }
    else {  
        m_self_recursive
              = [ [ PrefixSumMetalObjCRecursive alloc ] initWithNumElements : num_elements
                                                                       Type : algo_type
                                                             NumPartialSums : num_partial_sums
                                                                    ForFloat: true
                                                    NumThreadsPerThreadgroup: num_threads_per_threadgroup ];
    }
}


template<>
PrefixSumMetalCppImpl<int>::~PrefixSumMetalCppImpl(){;}

template<>
PrefixSumMetalCppImpl<float>::~PrefixSumMetalCppImpl(){;}

template<>
unsigned int PrefixSumMetalCppImpl<int>::numElements(unsigned int layer) {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw numElements ];
    }
    else {
        return [ m_self_recursive numElements:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<float>::numElements(unsigned int layer) {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw numElements ];
    }
    else {
        return [ m_self_recursive numElements:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<int>::numThreadsPerGroup( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ m_self_recursive numThreadsPerGroup:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<float>::numThreadsPerGroup( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ m_self_recursive numThreadsPerGroup:layer ];
    }
}

template<>
unsigned int PrefixSumMetalCppImpl<int>::numGroupsPerGrid( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ m_self_recursive numGroupsPerGrid:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<float>::numGroupsPerGrid( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ m_self_recursive numGroupsPerGrid:layer ];
    }
}


template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerIn() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw getRawPointerInForInt ];
    }
    else {
        return [ m_self_recursive getRawPointerInForInt ];
    }
}

template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerIn() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw getRawPointerInForFloat ];
    }
    else {
        return [ m_self_recursive getRawPointerInForFloat ];
    }
}

template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerOut() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw getRawPointerOutForInt ];
    }
    else {
        return [ m_self_recursive getRawPointerOutForInt ];
    }
}


template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerOut() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw getRawPointerOutForFloat ];
    }
    else {
        return [ m_self_recursive getRawPointerOutForFloat ];
    }
}


template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerGridPrefixSums( unsigned int layer, bool forIn ) {

    if ( m_algo_type == 3 ) {
        return nullptr;
    }
    else {
        return [ m_self_recursive getRawPointerGridPrefixSumsForInt:layer ForIn: forIn];
    }
}


template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerGridPrefixSums( unsigned int layer, bool forIn ) {

    if ( m_algo_type == 3 ) {
        return nullptr;
    }
    else {
        return [ m_self_recursive getRawPointerGridPrefixSumsForFloat:layer ForIn: forIn];
    }
}


template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerPartialSumsMerrillGrimshaw() {
    if ( m_algo_type == 3 ) {
         return [ m_self_merrill_grimshaw getRawPointerPartialSumsForInt ];
    }
    else {
        return nullptr;
    }
}


template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerPartialSumsMerrillGrimshaw() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw getRawPointerPartialSumsForFloat ];
    }
    else {
        return nullptr;
    }
}

template<>
void PrefixSumMetalCppImpl<int>::performComputation() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw performComputation ];
    }
    else {
        return [ m_self_recursive performComputation ];
    }
}


template<>
void PrefixSumMetalCppImpl<float>::performComputation() {
    if ( m_algo_type == 3 ) {
        return [ m_self_merrill_grimshaw performComputation ];
    }
    else {
        return [ m_self_recursive performComputation ];
    }
}


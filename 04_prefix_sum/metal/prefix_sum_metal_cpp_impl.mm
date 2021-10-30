#import "prefix_sum_metal_objc_recursive.h"
#import "prefix_sum_metal_objc_merrill_grimshaw.h"
#import "prefix_sum_metal_cpp_impl.h"

template<>
PrefixSumMetalCppImpl<int>::PrefixSumMetalCppImpl( const size_t num_elements, const int algo_type, const size_t num_partial_sums )
{
    m_algo_type = algo_type;

    if ( algo_type == 3 ) {

        m_self = [ [ PrefixSumMetalObjCMerrillGrimshaw alloc ] initWithNumElements : num_elements
                                                                   NumPartialSums : num_partial_sums 
                                                                          ForFloat: false ];
    }
    else {  
        m_self = [ [ PrefixSumMetalObjCRecursive alloc ] initWithNumElements : num_elements
                                                                       Type : algo_type
                                                             NumPartialSums : num_partial_sums 
                                                                   ForFloat : false ];
    }
}


template<>
PrefixSumMetalCppImpl<float>::PrefixSumMetalCppImpl( const size_t num_elements, const int algo_type, const size_t num_partial_sums )
{
    m_algo_type = algo_type;

    if ( algo_type == 3 ) {

        m_self = [ [ PrefixSumMetalObjCMerrillGrimshaw alloc ] initWithNumElements : num_elements
                                                                   NumPartialSums : num_partial_sums
                                                                          ForFloat: true ];
    }
    else {  
        m_self = [ [ PrefixSumMetalObjCRecursive alloc ] initWithNumElements : num_elements
                                                                       Type : algo_type
                                                             NumPartialSums : num_partial_sums
                                                                    ForFloat: true ];
    }
}


template<>
PrefixSumMetalCppImpl<int>::~PrefixSumMetalCppImpl(){;}


template<>
PrefixSumMetalCppImpl<float>::~PrefixSumMetalCppImpl(){;}


template<>
unsigned int PrefixSumMetalCppImpl<int>::numElements(unsigned int layer) {
    if ( m_algo_type == 3 ) {
        return [ (id)m_self numElements ];
    }
    else {
        return [ (id)m_self numElements:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<float>::numElements(unsigned int layer) {
    if ( m_algo_type == 3 ) {
        return [ (id)m_self numElements ];
    }
    else {
        return [ (id)m_self numElements:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<int>::numThreadsPerGroup( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ (id)m_self numThreadsPerGroup:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<float>::numThreadsPerGroup( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ (id)m_self numThreadsPerGroup:layer ];
    }
}

template<>
unsigned int PrefixSumMetalCppImpl<int>::numGroupsPerGrid( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ (id)m_self numGroupsPerGrid:layer ];
    }
}


template<>
unsigned int PrefixSumMetalCppImpl<float>::numGroupsPerGrid( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return 0;
    }
    else {
        return [ (id)m_self numGroupsPerGrid:layer ];
    }
}


template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerIn() {
    return [ (id)m_self getRawPointerInForInt ];
}

template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerIn() {
    return [ (id)m_self getRawPointerInForFloat ];
}

template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerOut() {
    return [ (id)m_self getRawPointerOutForInt ];
}


template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerOut() {
    return [ (id)m_self getRawPointerOutForFloat ];
}


template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerGridPrefixSums( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return nullptr;
    }
    else {
        return [ (id)m_self getRawPointerGridPrefixSumsForInt:layer ];
    }
}


template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerGridPrefixSums( unsigned int layer ) {

    if ( m_algo_type == 3 ) {
        return nullptr;
    }
    else {
        return [ (id)m_self getRawPointerGridPrefixSumsForFloat:layer ];
    }
}


template<>
int* PrefixSumMetalCppImpl<int>::getRawPointerPartialSumsMerrillGrimshaw() {
    return [ (id)m_self getRawPointerPartialSumsForInt ];
}


template<>
float* PrefixSumMetalCppImpl<float>::getRawPointerPartialSumsMerrillGrimshaw() {
    return [ (id)m_self getRawPointerPartialSumsForFloat ];
}

template<>
void PrefixSumMetalCppImpl<int>::performComputation() {
    return [ (id)m_self performComputation ];
}


template<>
void PrefixSumMetalCppImpl<float>::performComputation() {
    return [ (id)m_self performComputation ];
}


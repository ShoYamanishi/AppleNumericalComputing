#import "radix_sort_metal_objc.h"
#import "radix_sort_metal_cpp_impl.h"
                       
RadixSortMetalCppImpl::RadixSortMetalCppImpl(
    const size_t num_elements,
    const bool   for_float,
    const bool   coalesced_write,
    const bool   early_out
) {
    m_self = [ [ RadixSortMetalObjC alloc ] initWithNumElements: num_elements 
                                                      forFloat: for_float    
                                                CoalescedWrite: coalesced_write
                                                      EarlyOut: early_out ];
}

RadixSortMetalCppImpl::~RadixSortMetalCppImpl(){;}

void RadixSortMetalCppImpl::resetBufferFlag() {
    [ (id)m_self resetBufferFlag ];
}

unsigned int RadixSortMetalCppImpl::numElements() {
    return [ (id)m_self numElements ];
}

int* RadixSortMetalCppImpl::getRawPointerIn() {
    return [ (id)m_self getRawPointerIn ];
}

int* RadixSortMetalCppImpl::getRawPointerOut() {
    return [ (id)m_self getRawPointerOut ];
}

int* RadixSortMetalCppImpl::getRawPointerIn1() {
    return [ (id)m_self getRawPointerIn1 ];
}

int* RadixSortMetalCppImpl::getRawPointerIn2() {
    return [ (id)m_self getRawPointerIn2 ];
}

void RadixSortMetalCppImpl::performComputation() {
    return [ (id)m_self performComputation ];
}


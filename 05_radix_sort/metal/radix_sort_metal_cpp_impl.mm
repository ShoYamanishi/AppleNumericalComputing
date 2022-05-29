#import "radix_sort_metal_objc.h"
#import "radix_sort_metal_cpp_impl.h"
                       
RadixSortMetalCppImpl::RadixSortMetalCppImpl(
    const size_t num_elements,
    const bool   for_float,
    const bool   coalesced_write,
    const bool   early_out,
    const bool   in_one_commit,
    const size_t num_threads_per_threadgroup
) {

    m_self = [ [ RadixSortMetalObjC alloc ] initWithNumElements: num_elements 
                                                       forFloat: for_float    
                                                 CoalescedWrite: coalesced_write
                                                       EarlyOut: early_out
                                                    InOneCommit: in_one_commit
                                      NumThreadsPerThreadgrouop: num_threads_per_threadgroup ];
}

RadixSortMetalCppImpl::~RadixSortMetalCppImpl(){;}

void RadixSortMetalCppImpl::resetBufferFlag() {
    [ m_self resetBufferFlag ];
}

unsigned int RadixSortMetalCppImpl::numElements() {
    return [ m_self numElements ];
}

int* RadixSortMetalCppImpl::getRawPointerIn() {
    return [ m_self getRawPointerIn ];
}

int* RadixSortMetalCppImpl::getRawPointerOut() {
    return [ m_self getRawPointerOut ];
}

int* RadixSortMetalCppImpl::getRawPointerIn1() {
    return [ m_self getRawPointerIn1 ];
}

int* RadixSortMetalCppImpl::getRawPointerIn2() {
    return [ m_self getRawPointerIn2 ];
}

void RadixSortMetalCppImpl::performComputation() {
    return [ m_self performComputation ];
}

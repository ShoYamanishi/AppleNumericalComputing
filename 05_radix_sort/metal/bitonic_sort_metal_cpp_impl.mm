#import "bitonic_sort_metal_objc.h"
#import "bitonic_sort_metal_cpp_impl.h"
                       
BitonicSortMetalCppImpl::BitonicSortMetalCppImpl(
    const size_t num_elements,
    const bool   for_float,
    const size_t num_threads_per_threadgroup
) {
    m_self = [ [ BitonicSortMetalObjC alloc ] initWithNumElements: num_elements 
                                                         forFloat: for_float    
                                        NumThreadsPerThreadgrouop: num_threads_per_threadgroup ];
}

BitonicSortMetalCppImpl::~BitonicSortMetalCppImpl(){;}

unsigned int BitonicSortMetalCppImpl::numElements() {
    return [ m_self numElements ];
}

int* BitonicSortMetalCppImpl::getRawPointerInOut() {
    return [ m_self getRawPointerInOut ];
}

void BitonicSortMetalCppImpl::performComputation() {
    return [ m_self performComputation ];
}

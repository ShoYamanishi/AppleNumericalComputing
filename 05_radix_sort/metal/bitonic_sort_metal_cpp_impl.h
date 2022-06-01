#ifndef __BITONIC_SORT_METAL_CPP_IMPL_H__
#define __BITONIC_SORT_METAL_CPP_IMPL_H__

#include <cstddef>

class BitonicSortMetalCppImpl
{

  public:
    BitonicSortMetalCppImpl(
        const size_t num_elements, 
        const bool   for_float, 
        const size_t num_threads_per_threadgroup );

    virtual ~BitonicSortMetalCppImpl();

    unsigned int numElements();

    int* getRawPointerInOut();

    void performComputation();

  private:
    BitonicSortMetalObjC* m_self;

};

#endif /*__BITONIC_SORT_METAL_CPP_IMPL_H__*/

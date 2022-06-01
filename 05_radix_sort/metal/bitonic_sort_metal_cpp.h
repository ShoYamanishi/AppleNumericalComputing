#ifndef __BITONIC_SORT_METAL_CPP_H__
#define __BITONIC_SORT_METAL_CPP_H__

class BitonicSortMetalCppImpl;

class BitonicSortMetalCpp
{

  public:
    BitonicSortMetalCpp(
        const size_t num_elements, 
        const bool   for_float, 
        const size_t num_threads_per_threadgroup );

    virtual ~BitonicSortMetalCpp();

    unsigned int numElements();

    int* getRawPointerInOut();

    void performComputation();

  private:
    BitonicSortMetalCppImpl* m_impl;

};

#endif /*__BITONIC_SORT_METAL_CPP_H__*/

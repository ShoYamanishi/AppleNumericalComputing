#ifndef __RADIX_SORT_METAL_CPP_H__
#define __RADIX_SORT_METAL_CPP_H__

class RadixSortMetalCppImpl;

class RadixSortMetalCpp
{

  public:
    RadixSortMetalCpp(
        const size_t num_elements , 
        const bool   for_float, 
        const bool   coalesced_write, 
        const bool   early_out , 
        const bool   in_one_commit,
        const size_t num_threads_per_threadgroup );

    virtual ~RadixSortMetalCpp();

    void resetBufferFlag();

    unsigned int numElements();

    int* getRawPointerIn();

    int* getRawPointerOut();

    int* getRawPointerIn1();

    int* getRawPointerIn2();

    void performComputation();

  private:
    RadixSortMetalCppImpl* m_impl;

};

#endif /*__RADIX_SORT_METAL_CPP_H__*/

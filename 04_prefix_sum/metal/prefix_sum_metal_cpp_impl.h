#ifndef __PARTIAL_SUM_METAL_CPP_IMPL_H__
#define __PARTIAL_SUM_METAL_CPP_IMPL_H__

#include <cstddef>

template<class T>
class PrefixSumMetalCppImpl
{
    int m_algo_type;

  public:
    PrefixSumMetalCppImpl( const size_t num_elements, const int algo_type, const size_t num_partial_sums, const int num_threads_per_threadgroup );
    virtual ~PrefixSumMetalCppImpl();

    unsigned int numElements(unsigned int layer);

    unsigned int numThreadsPerGroup(unsigned int layer);

    unsigned int numGroupsPerGrid(unsigned int layer);

    T* getRawPointerIn();

    T* getRawPointerOut();

    T* getRawPointerGridPrefixSums(unsigned int layer, bool forIn );

    T* getRawPointerPartialSumsMerrillGrimshaw();

    void performComputation();

  private:
    PrefixSumMetalObjCRecursive*       m_self_recursive;
    PrefixSumMetalObjCMerrillGrimshaw* m_self_merrill_grimshaw;

};

#endif /*__PARTIAL_SUM_METAL_CPP_IMPL_H__*/

#ifndef __PREFIX_SUM_METAL_CPP_H__
#define __PREFIX_SUM_METAL_CPP_H__

template<class T>
class PrefixSumMetalCppImpl;

template<class T>
class PrefixSumMetalCpp
{

  public:
    PrefixSumMetalCpp( const size_t num_elements , const int algo_type, const size_t num_partial_sums );

    virtual ~PrefixSumMetalCpp();

    unsigned int numElements(unsigned int layer);

    unsigned int numThreadsPerGroup(unsigned int layer);

    unsigned int numGroupsPerGrid(unsigned int layer);

    T* getRawPointerIn();

    T* getRawPointerOut();

    T* getRawPointerGridPrefixSums(unsigned int layer);

    T* getRawPointerPartialSumsMerrillGrimshaw();

    void performComputation();

  private:
    PrefixSumMetalCppImpl<T>* m_impl;

};

#endif /*__PREFIX_SUM_METAL_CPP_H__*/

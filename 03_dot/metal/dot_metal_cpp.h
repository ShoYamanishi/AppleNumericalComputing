#ifndef __DOT_METAL_CPP_H__
#define __DOT_METAL_CPP_H__

class DotMetalCppImpl;

class DotMetalCpp
{

  public:

    // Reduction Type 1: Two-pass (baseline) :  Type 2 but without threadgroup memory (shared memory)
    // Reduction Type 2: Two-pass (shared mem) : Corresponds to 12.2 Two-Pass Reduction of The CUDA Handbook.
    // Reduction Type 3: Two-pass with SIMD shuffle
    // Reduction Type 4: One-pass atomic partial sum with SIMD shuffle : Similar to 12.4 Reduction with Atomics
    // Reduction Type 5: One-pass with SIMD add
    // Reduction Type 6: One-pass threadgroup counter partial sum (does not work for Metal) : Corresponds to 12.3 Single-Pass Reduction

    DotMetalCpp( const size_t num_elements , const size_t num_threads_per_group, const size_t num_groups_per_grid , const int reduction_type );

    virtual ~DotMetalCpp();

    float* getRawPointerX();

    float* getRawPointerY();

    float  getDotXY();

    void performComputation();

  private:
    DotMetalCppImpl* m_impl;

};

#endif /*__DOT_METAL_CPP_H__*/

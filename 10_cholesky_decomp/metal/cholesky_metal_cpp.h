#ifndef __CHOLESKY_METAL_CPP_H__
#define __CHOLESKY_METAL_CPP_H__

#include <simd/simd.h>
typedef unsigned int uint;

struct cholesky_constants {
    uint  dim;
};

class CholeskyMetalCppImpl;

class CholeskyMetalCpp
{

  public:
    CholeskyMetalCpp( const int dim, const bool use_mps );

    virtual ~CholeskyMetalCpp();

    float* getRawPointerL();

    float* getRawPointerX();

    float* getRawPointerY();

    float* getRawPointerB();

    void   setInitialStates( float* L, float* b );

    void   performComputation();

  private:
    CholeskyMetalCppImpl* m_impl;

};

#endif /*__CHOLESKY__METAL_CPP_H__*/

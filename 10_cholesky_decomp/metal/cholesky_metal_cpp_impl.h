#ifndef __CHOLESKY_METAL_CPP_IMPL_H__
#define __CHOLESKY_METAL_CPP_IMPL_H__

#include <cholesky_metal_cpp.h>

#include <cstddef>

class CholeskyMetalCppImpl
{
  public:
    CholeskyMetalCppImpl( const int dim, const bool useMPS );

    virtual ~CholeskyMetalCppImpl();

    float* getRawPointerL();

    float* getRawPointerX();

    float* getRawPointerY();

    float* getRawPointerB();

    void   setInitialStates( float* L, float* b );

    void   performComputation();

  private:
    void * m_self;
};

#endif /*__CHOLESKY_METAL_CPP_IMPL_H__*/

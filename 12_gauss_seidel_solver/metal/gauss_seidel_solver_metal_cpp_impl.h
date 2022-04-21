#ifndef __GAUSS_SEIDEL_SOLVER_METAL_CPP_IMPL_H__
#define __GAUSS_SEIDEL_SOLVER_METAL_CPP_IMPL_H__

#include "gauss_seidel_solver_metal_cpp.h"

#include <cstddef>

class GaussSeidelSolverMetalCppImpl
{

  public:
    GaussSeidelSolverMetalCppImpl( const int dim, const int iteration, const bool one_commit );

    virtual ~GaussSeidelSolverMetalCppImpl();

    float* getRawPointerA();

    float* getRawPointerB();

    float* getRawPointerActiveX();

    float  getError();

    void   setInitialStates( float* A, float* D, float* b, float* x1, float* x2 );

    void   performComputation();

  private:
    GaussSeidelSolverMetalObjC* m_self;
};

#endif /*__GAUSS_SEIDEL_SOLVER_METAL_CPP_IMPL_H__*/

#ifndef __JACOBI_SOLVER_METAL_CPP_IMPL_H__
#define __JACOBI_SOLVER_METAL_CPP_IMPL_H__

#include "jacobi_solver_metal_cpp.h"

#include <cstddef>

class JacobiSolverMetalCppImpl
{

  public:
    JacobiSolverMetalCppImpl( const int dim, const int iteration, const int type, const bool one_commit );

    virtual ~JacobiSolverMetalCppImpl();

    float* getRawPointerA();

    float* getRawPointerB();

    float* getRawPointerActiveX();

    float  getError();

    void   setInitialStates( float* A, float* D, float* b, float* x1, float* x2 );

    void   performComputation();

  private:
    JacobiSolverMetalObjC* m_self;
};

#endif /*__JACOBI_SOLVER_METAL_CPP_IMPL_H__*/

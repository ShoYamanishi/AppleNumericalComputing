#ifndef __JACOBI_SOLVER_METAL_CPP_H__
#define __JACOBI_SOLVER_METAL_CPP_H__

#include <simd/simd.h>

struct jacobi_solver_constants {
    int  dim;
};

class JacobiSolverMetalCppImpl;

class JacobiSolverMetalCpp
{

  public:
    JacobiSolverMetalCpp( const int dim, const int iteration, const int type );

    virtual ~JacobiSolverMetalCpp();

    float* getRawPointerA();

    float* getRawPointerB();

    float* getRawPointerActiveX();

    float  getError();

    void   setInitialStates( float* A, float* D, float* b, float* x1, float* x2 );

    void   performComputation();

  private:
    JacobiSolverMetalCppImpl* m_impl;

};

#endif /*__JACOBI_SOLVER_METAL_CPP_H__*/

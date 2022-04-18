#ifndef __GAUSS_SEIDEL_SOLVER_METAL_CPP_H__
#define __GAUSS_SEIDEL_SOLVER_METAL_CPP_H__

#include <simd/simd.h>

struct gauss_seidel_solver_constants {
    int  dim;
};

class GaussSeidelSolverMetalCppImpl;

class GaussSeidelSolverMetalCpp
{

  public:
    GaussSeidelSolverMetalCpp( const int dim, const int iteration, const bool one_commit );

    virtual ~GaussSeidelSolverMetalCpp();

    float* getRawPointerA();

    float* getRawPointerB();

    float* getRawPointerActiveX();

    float  getError();

    void   setInitialStates( float* A, float* D, float* b, float* x1, float* x2 );

    void   performComputation();

  private:
    GaussSeidelSolverMetalCppImpl* m_impl;

};

#endif /*__GAUSS_SEIDEL_SOLVER_METAL_CPP_H__*/

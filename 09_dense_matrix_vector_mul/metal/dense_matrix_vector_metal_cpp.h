#ifndef __DENSE_MATRIX_VECTOR_METAL_CPP_H__
#define __DENSE_MATRIX_VECTOR_METAL_CPP_H__

#include <simd/simd.h>

struct dense_matrix_vector_constants {
    int  M;
    int  N;
};

class DenseMatrixVectorMetalCppImpl;

class DenseMatrixVectorMetalCpp
{

  public:

    DenseMatrixVectorMetalCpp( const int M, const int N, const bool col_major, const bool threads_over_rows );

    DenseMatrixVectorMetalCpp( const int M, const int N ); // MPS

    virtual ~DenseMatrixVectorMetalCpp();

    float* getRawPointerMat();

    float* getRawPointerVec();

    float* getRawPointerOutVec();

    void   setInitialStates( float* M, float* v );

    void   performComputation();

  private:
    DenseMatrixVectorMetalCppImpl* m_impl;

};

#endif /*__DENSE_MATRIX_VECTOR_METAL_CPP_H__*/

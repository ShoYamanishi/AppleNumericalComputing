#ifndef __DENSE_MATRIX_VECTOR_METAL_CPP_IMPL_H__
#define __DENSE_MATRIX_VECTOR_METAL_CPP_IMPL_H__

#include <dense_matrix_vector_metal_cpp.h>

#include <cstddef>

class DenseMatrixVectorMetalCppImpl
{

  public:
    DenseMatrixVectorMetalCppImpl( const int M, const int N, const bool col_major, const bool threadss_over_rows );

    DenseMatrixVectorMetalCppImpl( const int M, const int N ); // MPS

    virtual ~DenseMatrixVectorMetalCppImpl();

    float* getRawPointerMat();

    float* getRawPointerVec();

    float* getRawPointerOutVec();

    void   setInitialStates( float* M, float* v );

    void   performComputation();

  private:
    void * m_self;
};

#endif /*__DENSE_MATRIX_VECTOR_METAL_CPP_IMPL_H__*/

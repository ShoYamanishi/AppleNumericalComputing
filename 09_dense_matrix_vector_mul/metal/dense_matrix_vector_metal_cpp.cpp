#include "dense_matrix_vector_metal_cpp_impl.h"
#include "dense_matrix_vector_metal_cpp.h"

DenseMatrixVectorMetalCpp::DenseMatrixVectorMetalCpp( const int M, const int N, const bool col_major, const bool threads_over_rows )
    :m_impl( new DenseMatrixVectorMetalCppImpl( M, N, col_major, threads_over_rows ) )
{
    ;
}

DenseMatrixVectorMetalCpp::DenseMatrixVectorMetalCpp( const int M, const int N )
    :m_impl( new DenseMatrixVectorMetalCppImpl( M, N ) )
{
    ;
}

DenseMatrixVectorMetalCpp::~DenseMatrixVectorMetalCpp()
{
    delete m_impl;
}

float* DenseMatrixVectorMetalCpp::getRawPointerMat()
{
    return m_impl->getRawPointerMat();
}

float* DenseMatrixVectorMetalCpp::getRawPointerVec()
{
    return m_impl->getRawPointerVec();
}

float* DenseMatrixVectorMetalCpp::getRawPointerOutVec()
{
    return m_impl->getRawPointerOutVec();
}

void DenseMatrixVectorMetalCpp::setInitialStates( float* M, float* v )
{
    return m_impl->setInitialStates( M, v );
}

void DenseMatrixVectorMetalCpp::performComputation()
{
    m_impl->performComputation();
}

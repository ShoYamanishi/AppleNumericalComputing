#import "dense_matrix_vector_metal_objc_own_shader.h"
#import "dense_matrix_vector_metal_objc_mps.h"
#import "dense_matrix_vector_metal_cpp_impl.h"

DenseMatrixVectorMetalCppImpl::DenseMatrixVectorMetalCppImpl( const int m, const int n, const bool col_major, const bool threads_over_rows )
{
    m_self = [ [ DenseMatrixVectorMetalObjCOwnShader alloc ] initWithM: m N: n ColMajor: col_major ThreadsOverRows: threads_over_rows ];
}

DenseMatrixVectorMetalCppImpl::DenseMatrixVectorMetalCppImpl( const int m, const int n)
{
    m_self = [ [ DenseMatrixVectorMetalObjCMPS alloc ] initWithM: m N: n ]; // MPS
}             

DenseMatrixVectorMetalCppImpl::~DenseMatrixVectorMetalCppImpl(){ m_self = nullptr; }

void DenseMatrixVectorMetalCppImpl::setInitialStates( float* M, float* v )
{
    [ (id)m_self setInitialStatesMat: M Vec: v ];
}

float* DenseMatrixVectorMetalCppImpl::getRawPointerMat() {

    return [ (id)m_self getRawPointerMat ];
}

float* DenseMatrixVectorMetalCppImpl::getRawPointerVec() {

    return [ (id)m_self getRawPointerVec ];
}

float* DenseMatrixVectorMetalCppImpl::getRawPointerOutVec() {

    return [ (id)m_self getRawPointerOutVec ];
}

void DenseMatrixVectorMetalCppImpl::performComputation()
{
    [ (id)m_self performComputation ];
}

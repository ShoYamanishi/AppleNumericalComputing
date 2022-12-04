#import "dense_matrix_vector_metal_objc_own_shader.h"
#import "dense_matrix_vector_metal_objc_mps.h"
#import "dense_matrix_vector_metal_cpp_impl.h"

DenseMatrixVectorMetalCppImpl::DenseMatrixVectorMetalCppImpl( const int m, const int n, const bool col_major, const bool threads_over_rows )
    :m_own_shader(true)
{
    m_self_own_shader = [ [ DenseMatrixVectorMetalObjCOwnShader alloc ] initWithM: m N: n ColMajor: col_major ThreadsOverRows: threads_over_rows ];
}

DenseMatrixVectorMetalCppImpl::DenseMatrixVectorMetalCppImpl( const int m, const int n)
    :m_own_shader(false)
{
    m_self_mps = [ [ DenseMatrixVectorMetalObjCMPS alloc ] initWithM: m N: n ]; // MPS
}             

DenseMatrixVectorMetalCppImpl::~DenseMatrixVectorMetalCppImpl(){;}

void DenseMatrixVectorMetalCppImpl::setInitialStates( float* M, float* v )
{
    if (m_own_shader) {
        [ m_self_own_shader setInitialStatesMat: M Vec: v ];
    }
    else {
        [ m_self_mps setInitialStatesMat: M Vec: v ];
    }
}

float* DenseMatrixVectorMetalCppImpl::getRawPointerMat() {
    if (m_own_shader) {
        return [ m_self_own_shader getRawPointerMat ];
    }
    else {
        return [ m_self_mps getRawPointerMat ];
    }
}
        

float* DenseMatrixVectorMetalCppImpl::getRawPointerVec() {
    if (m_own_shader) {
        return [ m_self_own_shader getRawPointerVec ];
    }
    else {
        return [ m_self_mps getRawPointerVec ];
    }
}

float* DenseMatrixVectorMetalCppImpl::getRawPointerOutVec() {
    if (m_own_shader) {
        return [ m_self_own_shader getRawPointerOutVec ];
    }
    else {
        return [ m_self_mps getRawPointerOutVec ];
    }
}

void DenseMatrixVectorMetalCppImpl::performComputation()
{
    if (m_own_shader) {
        [ m_self_own_shader performComputation ];
    }
    else {
        [ m_self_mps performComputation ];
    }
}

#import "cholesky_metal_objc_own_shader.h"
#import "cholesky_metal_objc_mps.h"
#import "cholesky_metal_cpp_impl.h"

CholeskyMetalCppImpl::CholeskyMetalCppImpl( const int dim, const bool use_mps )
    :m_use_mps(use_mps)
{
    if ( use_mps ) {
        m_self_mps = [ [ CholeskyMetalObjCMPS alloc ] initWithDim: dim ];
    }
    else {
        m_self_own_shader = [ [ CholeskyMetalObjCOwnShader alloc ] initWithDim: dim ];
    }

}

CholeskyMetalCppImpl::~CholeskyMetalCppImpl(){ m_self_own_shader = nullptr; m_self_mps = nullptr; }

void CholeskyMetalCppImpl::setInitialStates( float* L, float* b )
{
    if ( m_use_mps ) {
        [ m_self_mps setInitialStatesL: L B: b ];
    }
    else {
        [ m_self_own_shader setInitialStatesL: L B: b ];
    }
}

float* CholeskyMetalCppImpl::getRawPointerL() {
    if ( m_use_mps ) {
        return [ m_self_mps getRawPointerL ];
    }
    else {
        return [ m_self_own_shader getRawPointerL ];
    }
}

float* CholeskyMetalCppImpl::getRawPointerX() {
    if ( m_use_mps ) {
        return [ m_self_mps getRawPointerX ];
    }
    else {
        return [ m_self_own_shader getRawPointerX ];
    }
}

float* CholeskyMetalCppImpl::getRawPointerY() {
    if ( m_use_mps ) {
        return [ m_self_mps getRawPointerY ];
    }
    else {
        return [ m_self_own_shader getRawPointerY ];
    }
}

float* CholeskyMetalCppImpl::getRawPointerB() {
    if ( m_use_mps ) {
        return [ m_self_mps getRawPointerB ];
    }
    else {
        return [ m_self_own_shader getRawPointerB ];
    }
}

void CholeskyMetalCppImpl::performComputation()
{
    if ( m_use_mps ) {
        [ m_self_mps performComputation ];
    }
    else {
        [ m_self_own_shader performComputation ];
    }
}

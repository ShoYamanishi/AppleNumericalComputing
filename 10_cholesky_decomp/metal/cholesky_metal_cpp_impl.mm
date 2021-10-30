#import "cholesky_metal_cpp_impl.h"

#import "cholesky_metal_objc_own_shader.h"
#import "cholesky_metal_objc_mps.h"

CholeskyMetalCppImpl::CholeskyMetalCppImpl( const int dim, const bool use_mps )
{
    if ( use_mps ) {
        m_self = [ [ CholeskyMetalObjCMPS alloc ] initWithDim: dim ];
    }
    else {
        m_self = [ [ CholeskyMetalObjCOwnShader alloc ] initWithDim: dim ];
    }
}             

CholeskyMetalCppImpl::~CholeskyMetalCppImpl(){ m_self = nullptr; }

void CholeskyMetalCppImpl::setInitialStates( float* L, float* b )
{
    [ (id)m_self setInitialStatesL: L B: b ];
}

float* CholeskyMetalCppImpl::getRawPointerL() {

    return [ (id)m_self getRawPointerL ];
}

float* CholeskyMetalCppImpl::getRawPointerX() {

    return [ (id)m_self getRawPointerX ];
}

float* CholeskyMetalCppImpl::getRawPointerY() {

    return [ (id)m_self getRawPointerY ];
}

float* CholeskyMetalCppImpl::getRawPointerB() {

    return [ (id)m_self getRawPointerB ];
}

void CholeskyMetalCppImpl::performComputation()
{
    [ (id)m_self performComputation ];
}

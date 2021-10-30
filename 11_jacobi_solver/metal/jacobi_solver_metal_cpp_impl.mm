#import "jacobi_solver_metal_objc.h"
#import "jacobi_solver_metal_cpp_impl.h"


JacobiSolverMetalCppImpl::JacobiSolverMetalCppImpl( const int dim, const int iteration, const int type )
{
    m_self = [ [ JacobiSolverMetalObjC alloc ] initWithDim: dim Iteration: iteration Type: type ];
}             

JacobiSolverMetalCppImpl::~JacobiSolverMetalCppImpl(){ m_self = nullptr; }

void JacobiSolverMetalCppImpl::setInitialStates( float* A, float* D, float* b, float* x1, float* x2 )
{
    [ (id)m_self setInitialStatesA: A D:D B: b X1: x1 X2: x2 ];
}

float* JacobiSolverMetalCppImpl::getRawPointerA() {

    return [ (id)m_self getRawPointerA ];
}

float* JacobiSolverMetalCppImpl::getRawPointerB() {

    return [ (id)m_self getRawPointerB ];
}

float* JacobiSolverMetalCppImpl::getRawPointerActiveX() {
    return [ (id)m_self getRawPointerActiveX ];
}

float JacobiSolverMetalCppImpl::getError() {

    return [ (id)m_self getError ];
}

void JacobiSolverMetalCppImpl::performComputation()
{
    [ (id)m_self performComputation ];
}



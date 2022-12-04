#import "jacobi_solver_metal_objc.h"
#import "jacobi_solver_metal_cpp_impl.h"


JacobiSolverMetalCppImpl::JacobiSolverMetalCppImpl( const int dim, const int iteration, const int type, const bool one_commit )
{
    m_self = [ [ JacobiSolverMetalObjC alloc ] initWithDim: dim Iteration: iteration Type: type OneCommit: one_commit ];
}             

JacobiSolverMetalCppImpl::~JacobiSolverMetalCppImpl(){;}

void JacobiSolverMetalCppImpl::setInitialStates( float* A, float* D, float* b, float* x1, float* x2 )
{
    [ m_self setInitialStatesA: A D:D B: b X1: x1 X2: x2 ];
}

float* JacobiSolverMetalCppImpl::getRawPointerA() {

    return [ m_self getRawPointerA ];
}

float* JacobiSolverMetalCppImpl::getRawPointerB() {

    return [ m_self getRawPointerB ];
}

float* JacobiSolverMetalCppImpl::getRawPointerActiveX() {
    return [ m_self getRawPointerActiveX ];
}

float JacobiSolverMetalCppImpl::getError() {

    return [ m_self getError ];
}

void JacobiSolverMetalCppImpl::performComputation()
{
    [ m_self performComputation ];
}



#import "gauss_seidel_solver_metal_objc.h"
#import "gauss_seidel_solver_metal_cpp_impl.h"

GaussSeidelSolverMetalCppImpl::GaussSeidelSolverMetalCppImpl( const int dim, const int iteration, const bool one_commit )
{
    m_self = [ [ GaussSeidelSolverMetalObjC alloc ] initWithDim: dim Iteration: iteration oneCommit: one_commit ];
}             

GaussSeidelSolverMetalCppImpl::~GaussSeidelSolverMetalCppImpl(){;}

void GaussSeidelSolverMetalCppImpl::setInitialStates( float* A, float* D, float* b, float* x1, float* x2 )
{
    [ m_self setInitialStatesA: A D:D B: b X1: x1 X2: x2 ];
}

float* GaussSeidelSolverMetalCppImpl::getRawPointerA() {

    return [ m_self getRawPointerA ];
}

float* GaussSeidelSolverMetalCppImpl::getRawPointerB() {

    return [ m_self getRawPointerB ];
}

float* GaussSeidelSolverMetalCppImpl::getRawPointerActiveX() {
    return [ m_self getRawPointerActiveX ];
}

float GaussSeidelSolverMetalCppImpl::getError() {

    return [ m_self getError ];
}

void GaussSeidelSolverMetalCppImpl::performComputation()
{
    [ m_self performComputation ];
}



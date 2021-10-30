#import "gauss_seidel_solver_metal_objc.h"
#import "gauss_seidel_solver_metal_cpp_impl.h"

GaussSeidelSolverMetalCppImpl::GaussSeidelSolverMetalCppImpl( const int dim, const int iteration )
{
    m_self = [ [ GaussSeidelSolverMetalObjC alloc ] initWithDim: dim Iteration: iteration ];
}             

GaussSeidelSolverMetalCppImpl::~GaussSeidelSolverMetalCppImpl(){ m_self = nullptr; }

void GaussSeidelSolverMetalCppImpl::setInitialStates( float* A, float* D, float* b, float* x1, float* x2 )
{
    [ (id)m_self setInitialStatesA: A D:D B: b X1: x1 X2: x2 ];
}

float* GaussSeidelSolverMetalCppImpl::getRawPointerA() {

    return [ (id)m_self getRawPointerA ];
}

float* GaussSeidelSolverMetalCppImpl::getRawPointerB() {

    return [ (id)m_self getRawPointerB ];
}

float* GaussSeidelSolverMetalCppImpl::getRawPointerActiveX() {
    return [ (id)m_self getRawPointerActiveX ];
}

float GaussSeidelSolverMetalCppImpl::getError() {

    return [ (id)m_self getError ];
}

void GaussSeidelSolverMetalCppImpl::performComputation()
{
    [ (id)m_self performComputation ];
}



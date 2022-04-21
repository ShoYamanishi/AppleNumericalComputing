class GaussSeidelSolverMetalObjC;
#include "gauss_seidel_solver_metal_cpp_impl.h"
#include "gauss_seidel_solver_metal_cpp.h"


GaussSeidelSolverMetalCpp::GaussSeidelSolverMetalCpp( const int dim, const int iteration, const bool one_commit )
    :m_impl( new GaussSeidelSolverMetalCppImpl( dim, iteration, one_commit ) )
{
    ;
}

GaussSeidelSolverMetalCpp::~GaussSeidelSolverMetalCpp()
{
    delete m_impl;
}

float* GaussSeidelSolverMetalCpp::getRawPointerA()
{
    return m_impl->getRawPointerA();
}

float* GaussSeidelSolverMetalCpp::getRawPointerB()
{
    return m_impl->getRawPointerB();
}

float* GaussSeidelSolverMetalCpp::getRawPointerActiveX()
{
    return m_impl->getRawPointerActiveX();
}

float GaussSeidelSolverMetalCpp::getError()
{
    return m_impl->getError();
}

void GaussSeidelSolverMetalCpp::setInitialStates( float* A, float* D, float* b, float* x1, float* x2 )
{
    return m_impl->setInitialStates( A, D, b, x1, x2 );
}

void GaussSeidelSolverMetalCpp::performComputation()
{
    m_impl->performComputation();
}

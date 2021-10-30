#include "jacobi_solver_metal_cpp_impl.h"
#include "jacobi_solver_metal_cpp.h"

JacobiSolverMetalCpp::JacobiSolverMetalCpp( const int dim, const int iteration, const int type )
    :m_impl( new JacobiSolverMetalCppImpl( dim, iteration, type ) )
{
    ;
}

JacobiSolverMetalCpp::~JacobiSolverMetalCpp()
{
    delete m_impl;
}

float* JacobiSolverMetalCpp::getRawPointerA()
{
    return m_impl->getRawPointerA();
}

float* JacobiSolverMetalCpp::getRawPointerB()
{
    return m_impl->getRawPointerB();
}

float* JacobiSolverMetalCpp::getRawPointerActiveX()
{
    return m_impl->getRawPointerActiveX();
}

float JacobiSolverMetalCpp::getError()
{
    return m_impl->getError();
}

void JacobiSolverMetalCpp::setInitialStates( float* A, float* D, float* b, float* x1, float* x2 )
{
    return m_impl->setInitialStates( A, D, b, x1, x2 );
}

void JacobiSolverMetalCpp::performComputation()
{
    m_impl->performComputation();
}

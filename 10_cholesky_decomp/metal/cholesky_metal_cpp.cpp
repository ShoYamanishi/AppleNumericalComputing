class CholeskyMetalObjCOwnShader;
class CholeskyMetalObjCMPS;
#include "cholesky_metal_cpp_impl.h"
#include "cholesky_metal_cpp.h"

CholeskyMetalCpp::CholeskyMetalCpp( const int dim, const bool use_mps )
    :m_impl( new CholeskyMetalCppImpl( dim, use_mps ) )
{
    ;
}

CholeskyMetalCpp::~CholeskyMetalCpp()
{
    delete m_impl;
}

float* CholeskyMetalCpp::getRawPointerL()
{
    return m_impl->getRawPointerL();
}

float* CholeskyMetalCpp::getRawPointerX()
{
    return m_impl->getRawPointerX();
}

float* CholeskyMetalCpp::getRawPointerY()
{
    return m_impl->getRawPointerY();
}

float* CholeskyMetalCpp::getRawPointerB()
{
    return m_impl->getRawPointerB();
}

void CholeskyMetalCpp::setInitialStates( float* L, float* b )
{
    return m_impl->setInitialStates( L, b );
}

void CholeskyMetalCpp::performComputation()
{
    m_impl->performComputation();
}

class ConjugateGradientMetalObjC;
#include "conjugate_gradient_metal_cpp_impl.h"
#include "conjugate_gradient_metal_cpp.h"

ConjugateGradientMetalCpp::ConjugateGradientMetalCpp( const int dim, const int num_threads_per_group, const int max_num_iterations, const float epsilon )
    :m_impl( new ConjugateGradientMetalCppImpl( dim, num_threads_per_group, max_num_iterations, epsilon ) )
{
    ;
}

ConjugateGradientMetalCpp::~ConjugateGradientMetalCpp()
{
    delete m_impl;
};

float* ConjugateGradientMetalCpp::getRawPointerA()
{
    return m_impl->getRawPointerA();
}

float* ConjugateGradientMetalCpp::getRawPointerB()
{
    return m_impl->getRawPointerB();
}

float* ConjugateGradientMetalCpp::getX()
{
    return m_impl->getX();
}

int ConjugateGradientMetalCpp::getNumIterations()
{
    return m_impl->getNumIterations();
}

void ConjugateGradientMetalCpp::performComputation()
{
    m_impl->performComputation();
}


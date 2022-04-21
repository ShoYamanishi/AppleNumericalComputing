class FFTMetalObjC;
#include "fft_metal_cpp_impl.h"
#include "fft_metal_cpp.h"

FFTMetalCpp::FFTMetalCpp()
    :m_impl( new FFTMetalCppImpl() )
{
    ;
}

FFTMetalCpp::~FFTMetalCpp()
{
    delete m_impl;
}

float* FFTMetalCpp::getRawPointerTimeRe()
{
    return m_impl->getRawPointerTimeRe();
}

float* FFTMetalCpp::getRawPointerTimeIm()
{
    return m_impl->getRawPointerTimeIm();
}

float* FFTMetalCpp::getRawPointerFreqRe()
{
    return m_impl->getRawPointerFreqRe();
}

float* FFTMetalCpp::getRawPointerFreqIm()
{
    return m_impl->getRawPointerFreqIm();
}

void FFTMetalCpp::setInitialStates( float* time_re, float* time_im )
{
    return m_impl->setInitialStates( time_re, time_im );
}

void FFTMetalCpp::performComputation()
{
    m_impl->performComputation();
}

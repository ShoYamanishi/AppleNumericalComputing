#import "fft_metal_objc.h"
#import "fft_metal_cpp_impl.h"

FFTMetalCppImpl::FFTMetalCppImpl()
{
    m_self = [ [ FFTMetalObjC alloc ] init ];
}             

FFTMetalCppImpl::~FFTMetalCppImpl(){ m_self = nullptr; }

void FFTMetalCppImpl::setInitialStates( float* time_re, float* time_im )
{
    [ (id)m_self setInitialStatesTimeRe: time_re TimeIm: time_im ];
}

float* FFTMetalCppImpl::getRawPointerTimeRe() {

    return [ (id)m_self getRawPointerTimeRe ];
}

float* FFTMetalCppImpl::getRawPointerTimeIm() {

    return [ (id)m_self getRawPointerTimeIm ];
}

float* FFTMetalCppImpl::getRawPointerFreqRe() {

    return [ (id)m_self getRawPointerFreqRe ];
}

float* FFTMetalCppImpl::getRawPointerFreqIm() {

    return [ (id)m_self getRawPointerFreqIm ];
}

void FFTMetalCppImpl::performComputation()
{
    [ (id)m_self performComputation ];
}



#import "fft_metal_objc.h"
#import "fft_metal_cpp_impl.h"

FFTMetalCppImpl::FFTMetalCppImpl()
{
    m_self = [ [ FFTMetalObjC alloc ] init ];
}             

FFTMetalCppImpl::~FFTMetalCppImpl(){ m_self = nullptr; }

void FFTMetalCppImpl::setInitialStates( float* time_re, float* time_im )
{
    [ m_self setInitialStatesTimeRe: time_re TimeIm: time_im ];
}

float* FFTMetalCppImpl::getRawPointerTimeRe() {

    return [ m_self getRawPointerTimeRe ];
}

float* FFTMetalCppImpl::getRawPointerTimeIm() {

    return [ m_self getRawPointerTimeIm ];
}

float* FFTMetalCppImpl::getRawPointerFreqRe() {

    return [ m_self getRawPointerFreqRe ];
}

float* FFTMetalCppImpl::getRawPointerFreqIm() {

    return [ m_self getRawPointerFreqIm ];
}

void FFTMetalCppImpl::performComputation()
{
    [ m_self performComputation ];
}



#ifndef __FFT_METAL_CPP_IMPL_H__
#define __FFT_METAL_CPP_IMPL_H__

#include <fft_metal_cpp.h>

#include <cstddef>

class FFTMetalCppImpl
{

  public:
    FFTMetalCppImpl();

    virtual ~FFTMetalCppImpl();

    float* getRawPointerTimeRe();

    float* getRawPointerTimeIm();

    float* getRawPointerFreqRe();

    float* getRawPointerFreqIm();

    void   setInitialStates( float* time_re, float* time_im );

    void   performComputation();

  private:
    void * m_self;
};

#endif /*__FFT_METAL_CPP_IMPL_H__*/

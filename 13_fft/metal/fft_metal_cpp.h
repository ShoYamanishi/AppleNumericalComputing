#ifndef __FFT_METAL_CPP_H__
#define __FFT_METAL_CPP_H__

#include <simd/simd.h>

class FFTMetalCppImpl;

class FFTMetalCpp
{

  public:
    FFTMetalCpp();

    virtual ~FFTMetalCpp();

    float* getRawPointerTimeRe();

    float* getRawPointerTimeIm();

    float* getRawPointerFreqRe();

    float* getRawPointerFreqIm();

    void   setInitialStates( float* time_re, float* time_im );

    void   performComputation();

  private:
    FFTMetalCppImpl* m_impl;

};

#endif /*__FFT_METAL_CPP_H__*/

#ifndef __CONVOLUTION_2D_CIIMAGE_CPP_H__
#define __CONVOLUTION_2D_CIIMAGE_CPP_H__

#include "convolution_2d_ciimage_cpp_impl.h"

class Convolution2D_CIImageCpp
{
  public:
    Convolution2D_CIImageCpp( const size_t width, const size_t height, const size_t kernel_size, const bool use_gpu );
    virtual ~Convolution2D_CIImageCpp();
    void copyToInputBuffer(const float* const p);
    void copyToKernelBuffer(const float* const p);
    float* getOutputImagePtr();
    void performConvolution();

  private:
    Convolution2D_CIImageCppImpl *m_impl;
};

#endif /*__CONVOLUTION_2D_CIIMAGE_CPP_H__*/

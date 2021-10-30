#ifndef __CONVOLUTION_2D_METAL_CPP_H__
#define __CONVOLUTION_2D_METAL_CPP_H__

#include <cstddef>

class Convolution2DMetalCppImpl;

class Convolution2DMetalCpp
{
  public:
    Convolution2DMetalCpp( const size_t width, const size_t height, const size_t kernel_size , const int algo_type);
    virtual ~Convolution2DMetalCpp();

    void   copyToInputBuffer ( const float* const p );
    void   copyToKernelBuffer( const float* const p );
    float* getOutputImagePtr();
    void   performConvolution();

  private:
    Convolution2DMetalCppImpl* m_impl;
};

#endif /*__CONVOLUTION_2D_METAL_CPP_H__*/

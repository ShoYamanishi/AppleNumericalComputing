#ifndef __CONVOLUTION_2D_METAL_CPP_IMPL_H__
#define __CONVOLUTION_2D_METAL_CPP_IMPL_H__

#include "convolution_2d_metal_cpp.h"

class Convolution2DMetalCppImpl
{
  public:
    Convolution2DMetalCppImpl( const size_t width, const size_t height, const size_t kernel_size, const int algo_type );
    virtual ~Convolution2DMetalCppImpl();
    void   copyToInputBuffer ( const float* const p );
    void   copyToKernelBuffer( const float* const p );
    float* getOutputImagePtr();
    void   performConvolution();

  private:
    const int                         m_algo_type;
    Convolution2DMetalObjCOwnShader*  m_self_own_shader;
    Convolution2DMetalObjCMPS*        m_self_mps;
};
#endif /*__CONVOLUTION_2D_METAL_CPP_IMPL_H__*/

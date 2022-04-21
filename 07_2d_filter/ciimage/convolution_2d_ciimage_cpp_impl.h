#ifndef __CONVOLUTION_2D_CIIMAGE_CPP_IMPL_H__
#define __CONVOLUTION_2D_CIIMAGE_CPP_IMPL_H__

#include <cstddef>

class Convolution2D_CIImageCppImpl
{

  public:
    Convolution2D_CIImageCppImpl( const size_t width, const size_t height, const size_t kernel_size, const bool use_gpu );

    virtual ~Convolution2D_CIImageCppImpl();

    void copyToInputBuffer ( const float* const p );
    void copyToKernelBuffer( const float* const p );
    float* getOutputImagePtr();
    void performConvolution();

  private:
    Convolution2D_CIImageObjc* m_self;
};

#endif /*__CONVOLUTION_2D_CIIMAGE_CPP_IMPL_H__*/

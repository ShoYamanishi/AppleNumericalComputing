class Convolution2D_CIImageObjc;
#include "convolution_2d_ciimage_cpp_impl.h"
#include "convolution_2d_ciimage_cpp.h"

Convolution2D_CIImageCpp::Convolution2D_CIImageCpp(
    const size_t width,
    const size_t height,
    const size_t kernel_size,
    const bool   use_gpu
)
    :m_impl( new  Convolution2D_CIImageCppImpl( width, height, kernel_size, use_gpu ) )
{
    ;
}

Convolution2D_CIImageCpp::~Convolution2D_CIImageCpp()
{
    delete m_impl;
}


void Convolution2D_CIImageCpp::copyToInputBuffer(const float* const p)
{
    m_impl->copyToInputBuffer(p);
}


void Convolution2D_CIImageCpp::copyToKernelBuffer(const float* const p)
{
    m_impl->copyToKernelBuffer(p);
}


float* Convolution2D_CIImageCpp::getOutputImagePtr()
{
    return m_impl->getOutputImagePtr();
}

void Convolution2D_CIImageCpp::performConvolution()
{
    m_impl->performConvolution();
}

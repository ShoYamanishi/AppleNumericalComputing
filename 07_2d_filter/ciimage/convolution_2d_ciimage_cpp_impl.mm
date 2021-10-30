#import "convolution_2d_ciimage_objc.h"
#import "convolution_2d_ciimage_cpp_impl.h"

Convolution2D_CIImageCppImpl::Convolution2D_CIImageCppImpl(
    const size_t width,
    const size_t height,
    const size_t kernel_size,
    const bool   use_gpu
) {

    m_self = [ [ Convolution2D_CIImageObjc alloc ] initWithWidth: width Height: height KernelSize: kernel_size UseGPU:use_gpu ];
}

Convolution2D_CIImageCppImpl::~Convolution2D_CIImageCppImpl()
{
    [ (id)m_self release_explicit];
}

void Convolution2D_CIImageCppImpl::copyToInputBuffer(const float* const p)
{
    [ (id)m_self copyToInputBuffer: p ];
}

void Convolution2D_CIImageCppImpl::copyToKernelBuffer(const float* const p)
{
    [ (id)m_self copyToKernelBuffer: p ];
}

float* Convolution2D_CIImageCppImpl::getOutputImagePtr()
{
    return [ (id)m_self getOutputImagePtr ];
}

void Convolution2D_CIImageCppImpl::performConvolution()
{
    [ (id)m_self performConvolution ];
}

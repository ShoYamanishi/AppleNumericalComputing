#import "convolution_2d_metal_objc_own_shader.h"
#import "convolution_2d_metal_objc_mps.h"
#import "convolution_2d_metal_cpp_impl.h"

Convolution2DMetalCppImpl::Convolution2DMetalCppImpl( const size_t width, const size_t height, const size_t kernel_size , const int algo_type )
{
    if ( algo_type == 0 ) {
        m_self = [ [ Convolution2DMetalObjCOwnShader alloc ] initWithWidth: width Height: height KernelSize: kernel_size Use2Stages: false ];
    }
    else if ( algo_type == 1 ) {
        m_self = [ [ Convolution2DMetalObjCOwnShader alloc ] initWithWidth: width Height: height KernelSize: kernel_size Use2Stages: true  ];
    }
    else {
        m_self = [ [ Convolution2DMetalObjCMPS       alloc ] initWithWidth: width Height: height KernelSize: kernel_size ];
    }
}

Convolution2DMetalCppImpl::~Convolution2DMetalCppImpl(){;}

void Convolution2DMetalCppImpl::copyToInputBuffer ( const float* const p )
{
    [ (id)m_self copyToInputBuffer:p ];
}

void Convolution2DMetalCppImpl::copyToKernelBuffer ( const float* const p )
{
    [ (id)m_self copyToKernelBuffer:p ];
}

float* Convolution2DMetalCppImpl::getOutputImagePtr()
{
    return [ (id)m_self getOutputImagePtr ];
}

void Convolution2DMetalCppImpl::performConvolution()
{
    [ (id)m_self performConvolution ];
}

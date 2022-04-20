#import "convolution_2d_metal_objc_own_shader.h"
#import "convolution_2d_metal_objc_mps.h"
#import "convolution_2d_metal_cpp_impl.h"

Convolution2DMetalCppImpl::Convolution2DMetalCppImpl( const size_t width, const size_t height, const size_t kernel_size , const int algo_type ): m_algo_type( algo_type )
{
    
    if ( algo_type == 0 ) {
        m_self_own_shader = [ [ Convolution2DMetalObjCOwnShader alloc ] initWithWidth: width Height: height KernelSize: kernel_size Use2Stages: false ];
    }
    else if ( algo_type == 1 ) {
        m_self_own_shader = [ [ Convolution2DMetalObjCOwnShader alloc ] initWithWidth: width Height: height KernelSize: kernel_size Use2Stages: true  ];
    }
    else {
        m_self_mps = [ [ Convolution2DMetalObjCMPS       alloc ] initWithWidth: width Height: height KernelSize: kernel_size ];
    }
}

Convolution2DMetalCppImpl::~Convolution2DMetalCppImpl(){;}

void Convolution2DMetalCppImpl::copyToInputBuffer ( const float* const p )
{
    if ( m_algo_type == 0 || m_algo_type == 1 ) {
        [ m_self_own_shader copyToInputBuffer:p ];
    }
    else {
        [ m_self_mps copyToInputBuffer:p ];
    }
}

void Convolution2DMetalCppImpl::copyToKernelBuffer ( const float* const p )
{
    if ( m_algo_type == 0 || m_algo_type == 1 ) {
        [ m_self_own_shader copyToKernelBuffer:p ];
    }
    else {
        [ m_self_mps copyToKernelBuffer:p ];
    }
}

float* Convolution2DMetalCppImpl::getOutputImagePtr()
{
    if ( m_algo_type == 0 || m_algo_type == 1 ) {
        return [ m_self_own_shader getOutputImagePtr ];
    }
    else {
        return [ m_self_mps getOutputImagePtr ];
    }
}
void Convolution2DMetalCppImpl::performConvolution()
{
    if ( m_algo_type == 0 || m_algo_type == 1 ) {
        [ m_self_own_shader performConvolution ];
    }
    else {
        [ m_self_mps performConvolution ];
    }
}

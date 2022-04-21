
class Convolution2DMetalObjCOwnShader;
class Convolution2DMetalObjCMPS;

#include "convolution_2d_metal_cpp_impl.h"
#include "convolution_2d_metal_cpp.h"

#include <type_traits>
#include <iostream>


Convolution2DMetalCpp::Convolution2DMetalCpp( const size_t width, const size_t height, const size_t kernel_size, const int algo_type )
    :m_impl( new Convolution2DMetalCppImpl( width, height, kernel_size, algo_type ) )
{
    ;
}

Convolution2DMetalCpp::~Convolution2DMetalCpp()
{
    delete m_impl;
};

void Convolution2DMetalCpp::copyToInputBuffer ( const float* const p )
{
    m_impl->copyToInputBuffer ( p );
}

void Convolution2DMetalCpp::copyToKernelBuffer ( const float* const p )
{
    m_impl->copyToKernelBuffer ( p );
}

float* Convolution2DMetalCpp::getOutputImagePtr()
{
    return m_impl->getOutputImagePtr();
}

void Convolution2DMetalCpp::performConvolution()
{
    m_impl->performConvolution();
}

#include <memory>
#include <algorithm>
#include <iostream>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef unsigned int uint;

#import "convolution_2d_metal_objc_mps.h"

@implementation Convolution2DMetalObjCMPS
{
    uint                        _mImageWidth;
    uint                        _mImageHeight;
    uint                        _mKernelSize;

    id<MTLDevice>               _mDevice;
    id<MTLCommandQueue>         _mCommandQueue;
    id<MTLTexture>              _mMPSTexture;

    float                       _mMPSWeights[ 7*7 ];
    float*                      _mMPSOutput;
}

- (instancetype) initWithWidth:(size_t) width Height:(size_t) height KernelSize:(size_t) kernel_size
{
    self = [super init];
    if (self)
    {

        _mImageWidth  = width;
        _mImageHeight = height;
        _mKernelSize  = kernel_size;

        _mDevice   = MTLCreateSystemDefaultDevice();
        if ( _mDevice == nil ) {
            NSLog(@"Failed to get the default Metal device.");
            return nil;
        }

        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }

        // MPSImageConvolution
        MTLTextureDescriptor *tex_desc = [[MTLTextureDescriptor alloc] init];
        if (tex_desc == nil) {
            NSLog(@"Failed to allocate MTLTextureDescriptor.");
            return nil;
        }
        tex_desc.width       = _mImageWidth;
        tex_desc.height      = _mImageHeight;
        tex_desc.pixelFormat = MTLPixelFormatR32Float; // Ordinary format with one 32-bit floating-point component.
        tex_desc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        tex_desc.textureType = MTLTextureType2D;
        _mMPSTexture = [ _mDevice newTextureWithDescriptor:tex_desc ];
        if (_mMPSTexture == nil) {
            NSLog(@"Failed to allocate _mMPSTexture.");
            return nil;
        }

        _mMPSOutput = (float*)malloc( sizeof(float) * _mImageWidth * _mImageHeight );
        if ( _mMPSOutput == nullptr ) {
            NSLog(@"Failed to allocate _mMPSOutput.");
            return nil;
        }
    }

    return self;
}


- (void) copyToInputBuffer: (const float* const) p
{
    MTLRegion region = MTLRegionMake2D( 0, 0, _mImageWidth, _mImageHeight );

    [ _mMPSTexture replaceRegion: region
                     mipmapLevel: 0
                       withBytes: p
                     bytesPerRow: sizeof(float)*_mImageWidth ];
}


- (void) copyToKernelBuffer:(const float* const) p
{
    memcpy( _mMPSWeights, p, sizeof(float) * _mKernelSize * _mKernelSize );
}


- (float*) getOutputImagePtr
{
    MTLRegion region = MTLRegionMake2D( 0, 0, _mImageWidth, _mImageHeight );

    [ _mMPSTexture getBytes: _mMPSOutput
                bytesPerRow: sizeof(float) * _mImageWidth
              bytesPerImage: sizeof(float) * _mImageWidth * _mImageHeight
                 fromRegion: region
                mipmapLevel: 0
                      slice: 0 ];

    return (float*)_mMPSOutput;
}

// from https://developer.apple.com/documentation/metalperformanceshaders/mpscopyallocator?language=objc

MPSCopyAllocator myAllocator
    = ^id <MTLTexture> __nonnull NS_RETURNS_RETAINED (
                        MPSKernel * __nonnull filter,
                        id <MTLCommandBuffer> __nonnull cmdBuf,
                        id <MTLTexture> __nonnull sourceTexture
) {

    MTLPixelFormat format = sourceTexture.pixelFormat;

    MTLTextureDescriptor* d
        = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: format
                                                             width: sourceTexture.width
                                                            height: sourceTexture.height
                                                         mipmapped: NO                   ];

    d.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.textureType = MTLTextureType2D;
    id <MTLTexture> result = [ cmdBuf.device newTextureWithDescriptor: d ];
 
    return result;    
   // d is autoreleased.
};


- (void) performConvolution
{

    id<MTLCommandBuffer> commandBuffer = [ _mCommandQueue commandBuffer ];

    MPSImageConvolution* MPSConvKernel
        = [ [MPSImageConvolution alloc] initWithDevice: _mDevice
                                           kernelWidth: _mKernelSize
                                          kernelHeight: _mKernelSize
                                               weights: _mMPSWeights ];

    [ MPSConvKernel encodeToCommandBuffer: commandBuffer inPlaceTexture: &_mMPSTexture fallbackCopyAllocator: myAllocator ];

    [ commandBuffer commit];

    [ commandBuffer waitUntilCompleted ];

    //[ MPSConvKernel release ];
}

@end

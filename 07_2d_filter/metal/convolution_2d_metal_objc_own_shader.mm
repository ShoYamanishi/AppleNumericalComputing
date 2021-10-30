#include <memory>
#include <algorithm>
#include <iostream>

typedef unsigned int uint;

struct convolution_2d_constants {
    uint num_elements;
    uint num_elements_stage_2;
    uint image_width;
    uint image_height;
    uint kernel_size;
};

#import "convolution_2d_metal_objc_own_shader.h"

static const int MAXIMUM_THREADS_PER_THREADGROUP_FOR_STAGE_2 = 768;

@implementation Convolution2DMetalObjCOwnShader
{
    id<MTLComputePipelineState> _mPSO_convolution_2d_naive;
    id<MTLComputePipelineState> _mPSO_convolution_5x5_stage1;
    id<MTLComputePipelineState> _mPSO_convolution_5x5_stage2;

    id<MTLBuffer>               _mInput;
    id<MTLBuffer>               _mOutput;
    id<MTLBuffer>               _mKernel;
    id<MTLBuffer>               _mConst;

    bool                        _mUse2Stages;

    uint                        _mImageWidth;
    uint                        _mImageHeight;
    uint                        _mKernelSize;

    uint                        _mNumElements;
    uint                        _mNumThreadsPerGroup;
    uint                        _mNumGroupsPerGrid;

    uint                        _mNumElementsStage2;
    uint                        _mNumThreadsPerGroupStage2;
    uint                        _mNumGroupsPerGridStage2;
}

- (instancetype) initWithWidth:(size_t) width Height:(size_t) height KernelSize:(size_t) kernel_size Use2Stages:(bool) use2stages
{
    self = [super init];
    if (self)
    {
        _mUse2Stages         = use2stages;
        _mNumElements        = width*height;
        _mNumThreadsPerGroup = 1024;
        _mNumGroupsPerGrid   = (_mNumElements + 1023) / 1024;

        if ( _mNumGroupsPerGrid == 1 && _mNumElements < 1024 ){

            _mNumThreadsPerGroup  = ((_mNumElements + 31) / 32) * 32;
        }

        _mNumElementsStage2        = _mNumElements / 8;
        _mNumThreadsPerGroupStage2 = MAXIMUM_THREADS_PER_THREADGROUP_FOR_STAGE_2;
        _mNumGroupsPerGridStage2   = (_mNumElementsStage2 + MAXIMUM_THREADS_PER_THREADGROUP_FOR_STAGE_2 -1 ) / MAXIMUM_THREADS_PER_THREADGROUP_FOR_STAGE_2;

        if ( _mNumGroupsPerGridStage2 == 1 && _mNumElementsStage2 <  MAXIMUM_THREADS_PER_THREADGROUP_FOR_STAGE_2 ){

            _mNumThreadsPerGroupStage2  = ((_mNumElementsStage2 + 31) / 32) * 32;
        }

        _mImageWidth  = width;
        _mImageHeight = height;
        _mKernelSize  = kernel_size;

        [ self loadLibraryWithName: @"./convolution_2d.metallib" ];

        _mPSO_convolution_2d_naive   = [ self getPipelineStateForFunction: @"convolution_2d_naive"   ];
        _mPSO_convolution_5x5_stage1 = [ self getPipelineStateForFunction: @"convolution_5x5_stage1" ];
        _mPSO_convolution_5x5_stage2 = [ self getPipelineStateForFunction: @"convolution_5x5_stage2" ];

        _mInput  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mImageWidth * _mImageHeight for: @"_mInput"  ];
        _mOutput = [ self getSharedMTLBufferForBytes: sizeof(float) * _mImageWidth * _mImageHeight for: @"_mOutput" ];
        _mKernel = [ self getSharedMTLBufferForBytes: sizeof(float) * _mKernelSize * _mKernelSize  for: @"_mKernel" ];
        _mConst  = [ self getSharedMTLBufferForBytes: sizeof(struct convolution_2d_constants)      for: @"_mConst"  ];

        struct convolution_2d_constants c;
        memset( &c, (uint)0, sizeof (struct convolution_2d_constants) );
        c. num_elements         = _mNumElements;
        c. num_elements_stage_2 = _mNumElementsStage2;
        c.image_width           = _mImageWidth;
        c.image_height          = _mImageHeight;
        c.kernel_size           = _mKernelSize;
        memcpy( _mConst.contents, &c, sizeof(struct convolution_2d_constants) );
    }
    return self;
}


- (void) copyToInputBuffer: (const float* const) p
{
    memcpy(_mInput.contents, p, sizeof(float) * _mImageWidth * _mImageHeight );
}


- (void) copyToKernelBuffer:(const float* const) p
{
    memcpy( _mKernel.contents, p, sizeof(float) * _mKernelSize * _mKernelSize );
}


- (float*) getOutputImagePtr
{
    return (float*)_mOutput.contents;
}


-(void)  performConvolution
{
    if ( _mUse2Stages ) {
        [ self performConvolution2Stages ];
    }
    else {
        [ self performConvolutionNaive ];
    }
}


- (void) performConvolutionNaive
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO_convolution_2d_naive ];

    [ computeEncoder setBuffer:_mInput                 offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mKernel                offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mOutput                offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:3 ];
    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}


- (void) performConvolution2Stages
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO_convolution_5x5_stage1 ];

    [ computeEncoder setBuffer:_mInput                 offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mKernel                offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mOutput                offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:3 ];
    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ computeEncoder setComputePipelineState: _mPSO_convolution_5x5_stage2 ];

    [ computeEncoder setBuffer:_mInput                 offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mKernel                offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mOutput                offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:3 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridStage2,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupStage2, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}


@end

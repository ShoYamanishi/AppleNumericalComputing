#include <memory>
#include <iostream>

#import "conjugate_gradient_metal_objc.h"

struct conjugate_gradient_constants {
    int   dim;
    float epsilon;
    int   max_num_iterations;
};


@implementation ConjugateGradientMetalObjC
{
    id<MTLComputePipelineState> _mPSO;

    id<MTLBuffer> _mA;
    id<MTLBuffer> _mX;
    id<MTLBuffer> _mB;
    id<MTLBuffer> _mR;
    id<MTLBuffer> _mP;
    id<MTLBuffer> _mAP;
    id<MTLBuffer> _mNumIterations;
    id<MTLBuffer> _mConf;
    uint          _mDim;
    uint          _mNumThreadsPerGroup;
}

- (instancetype) initWithNumElements: (int)   dim
                  NumThreadsPerGroup: (int)   num_threads_per_group
                    MaxNumIterations: (int)   max_num_iterations
                             Epsilon: (float) epsilon
{
    self = [super init];
    if (self)
    {
        _mDim                = dim;
        _mNumThreadsPerGroup = num_threads_per_group;

        [ self loadLibraryWithName:@"./conjugate_gradient.metallib" ];

        _mPSO = [ self getPipelineStateForFunction:@"conjugate_gradient" ];

        _mA   = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim * _mDim for:@"_mA" ];
        _mX   = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for:@"_mX" ];
        _mB   = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for:@"_mB" ];
        _mR   = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for:@"_mR" ];
        _mP   = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for:@"_mP" ];
        _mAP  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for:@"_mAP" ];

        _mNumIterations = [ self getSharedMTLBufferForBytes: sizeof(int) for:@"_mNumIterations" ];
        _mConf = [ self getSharedMTLBufferForBytes: sizeof(struct conjugate_gradient_constants) for:@"_mConf" ];

        struct conjugate_gradient_constants c;

        memset( &c, (int)0, sizeof(struct conjugate_gradient_constants) );
        c.dim     = dim;
        c.epsilon = epsilon;
        c.max_num_iterations = max_num_iterations;
        memcpy( _mConf.contents, &c, sizeof(struct conjugate_gradient_constants) );
    }
    return self;
}

-(float*) getRawPointerA
{
    return (float*)_mA.contents;
}

-(float*) getRawPointerB
{
    return (float*)_mB.contents;
}

-(float*) getX
{
    return (float*)_mX.contents;
}

- (int) getNumIterations
{
    return ((int*)_mNumIterations.contents)[0];
}

-(void) performComputation
{
    memset( _mX.contents, 0, sizeof(float) * _mDim );

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState:_mPSO ];

    [ computeEncoder setBuffer:_mA             offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mX             offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mB             offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mR             offset:0 atIndex:3 ];
    [ computeEncoder setBuffer:_mP             offset:0 atIndex:4 ];
    [ computeEncoder setBuffer:_mAP            offset:0 atIndex:5 ];
    [ computeEncoder setBuffer:_mConf          offset:0 atIndex:6 ];
    [ computeEncoder setBuffer:_mNumIterations offset:0 atIndex:7 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( 1, 1, 1 )
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1 ) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

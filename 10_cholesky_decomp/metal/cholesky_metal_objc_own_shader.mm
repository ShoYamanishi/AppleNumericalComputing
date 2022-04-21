#include <memory>
#include <algorithm>
#include <iostream>

#import "cholesky_metal_objc_own_shader.h"

@implementation CholeskyMetalObjCOwnShader
{
    uint                        _mDim;

    id<MTLDevice>               _mDevice;
    id<MTLComputePipelineState> _mPSO_decompose_cholesky;
    id<MTLComputePipelineState> _mPSO_solve_Lyeb;
    id<MTLComputePipelineState> _mPSO_solve_Ltxey;

    id<MTLCommandQueue>         _mCommandQueue;

    id<MTLBuffer>               _mL;
    id<MTLBuffer>               _mx;
    id<MTLBuffer>               _my;
    id<MTLBuffer>               _mb;
    id<MTLBuffer>               _mConst;

}


- (instancetype) initWithDim:(size_t) dim
{
    self = [super init];

    if (self)
    {
        _mDim = dim;

        [ self loadLibraryWithName: @"./cholesky.metallib" ];

        _mPSO_decompose_cholesky = [ self getPipelineStateForFunction: @"decompose_cholesky" ];
        _mPSO_solve_Lyeb         = [ self getPipelineStateForFunction: @"solve_Lyeb"         ];
        _mPSO_solve_Ltxey        = [ self getPipelineStateForFunction: @"solve_Ltxey"        ];

        _mL     = [ self getSharedMTLBufferForBytes: sizeof(float) * (_mDim + 1) * _mDim / 2 for: @"_mL"     ];
        _mx     = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim                   for: @"_mx"     ];
        _my     = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim                   for: @"_my"     ];
        _mb     = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim                   for: @"_mb"     ];

        _mConst = [ self getSharedMTLBufferForBytes: sizeof(struct cholesky_constants)       for: @"_mConst" ];
        struct cholesky_constants c;
        memset( &c, (uint)0, sizeof(struct cholesky_constants) );
        c.dim = _mDim;
        memcpy( _mConst.contents, &c, sizeof(struct cholesky_constants) );
    }
    return self;
}


- (void) setInitialStatesL:(float*) L B:(float*) b
{
    memcpy( _mL.contents,   L,   (_mDim + 1) * _mDim * sizeof(float) / 2 );
    memcpy( _mb.contents,   b,   _mDim * sizeof(float) );
}


- (float*) getRawPointerL
{
    return (float*)_mL.contents;
}


- (float*) getRawPointerX
{
    return (float*)_mx.contents;
}


- (float*) getRawPointerY
{
    return (float*)_my.contents;
}


- (float*) getRawPointerB
{
    return (float*)_mb.contents;
}


- (void) performComputation
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO_decompose_cholesky ];

    [ computeEncoder setBuffer:_mL     offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mConst offset:0 atIndex:1 ];
    
    int numGroupsPerGrid   = 1;
    int numThreadsPerGroup =   (_mDim>=1024)
                             ? 1024
                             : ( ((_mDim + 31) / 32) * 32 ) ;

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( numThreadsPerGroup, 1, 1) ];

    [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ computeEncoder setComputePipelineState: _mPSO_solve_Lyeb ];

    [ computeEncoder setBuffer:_mL             offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_my             offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mb             offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst         offset:0 atIndex:3 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( 1024, 1, 1) ];

    [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ computeEncoder setComputePipelineState: _mPSO_solve_Ltxey ];

    [ computeEncoder setBuffer:_mL             offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mx             offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_my             offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst         offset:0 atIndex:3 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( 1024, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}


@end

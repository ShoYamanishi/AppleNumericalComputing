#include <memory>
#include <algorithm>
#include <iostream>

#import "gauss_seidel_solver_metal_objc.h"

@implementation GaussSeidelSolverMetalObjC
{
    int _mDim;
    int _mIteration;
    bool _mX1ToX2;

    id<MTLComputePipelineState> _mPSO_solve_raw_major;

    id<MTLBuffer>               _mA;
    id<MTLBuffer>               _mB;
    id<MTLBuffer>               _mDinv;
    id<MTLBuffer>               _mX1;
    id<MTLBuffer>               _mX2;
    id<MTLBuffer>               _mError;
    id<MTLBuffer>               _mConst;
}


- (instancetype)initWithDim:(int) dim Iteration:(int) iteration
{

    self = [super init];

    if (self)
    {
        _mDim       = dim;
        _mIteration = iteration;
        _mX1ToX2    = false;

        [ self loadLibraryWithName: @"./gauss_seidel_solver.metallib" ];

        _mPSO_solve_raw_major =  [ self getPipelineStateForFunction: @"solve_raw_major" ];

        _mA     = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim * _mDim for: @"_mA"    ];
        _mB     = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for: @"_mB"    ];
        _mDinv  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for: @"_mDinv" ];
        _mX1    = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for: @"_mX1"   ];
        _mX2    = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim         for: @"_mX2"   ];
        _mError = [ self getSharedMTLBufferForBytes: sizeof(float)                 for: @"_mError"];
        _mConst = [ self getSharedMTLBufferForBytes: sizeof(struct gauss_seidel_solver_constants)  for: @"_mConst"];
        struct gauss_seidel_solver_constants c;
        memset( &c, (uint)0, sizeof(struct gauss_seidel_solver_constants) );
        c.dim = _mDim;
        memcpy( _mConst.contents, &c, sizeof(struct gauss_seidel_solver_constants) );
    }
    return self;
}


- (void) setInitialStatesA:(float*) A D:(float*) D B:(float*) b X1:(float*) x1 X2:(float*) x2
{
    memcpy( _mA.contents, A, _mDim * _mDim * sizeof(float) );
    memcpy( _mB.contents, b, _mDim * sizeof(float) );

    for ( int i = 0; i < _mDim; i++ ) {
        ((float*)_mDinv.contents)[i] = 1.0 / D[i];
    }
    memcpy( _mX1.contents, x1, _mDim * sizeof(float) );
    memcpy( _mX2.contents, x2, _mDim * sizeof(float) );
}

- (float*) getRawPointerA
{
    return (float*)_mA.contents;
}

- (float*) getRawPointerB
{
    return (float*)_mB.contents;
}

- (float*) getRawPointerActiveX
{
    if ( _mX1ToX2 ) {
        return (float*)_mX2.contents;
    }
    else {
        return (float*)_mX1.contents;
    }
}

- (float) getError
{
    return ((float*)(_mError.contents))[0];
}


- (void) performComputation
{
    memset( _mX1.contents, 0, sizeof(float) * _mDim );
    memset( _mX2.contents, 0, sizeof(float) * _mDim );

    _mX1ToX2    = false;

    for ( int i = 0; i < _mIteration; i++ ) {

        id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

        assert( commandBuffer != nil );

        id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

        assert( computeEncoder != nil );

        [ computeEncoder setComputePipelineState: _mPSO_solve_raw_major ];

        [ computeEncoder setBuffer:_mA       offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mDinv    offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mB       offset:0 atIndex:2 ];

        if ( _mX1ToX2 ) {
            [ computeEncoder setBuffer:_mX1      offset:0 atIndex:3 ];
            [ computeEncoder setBuffer:_mX2      offset:0 atIndex:4 ];
        }
        else {
            [ computeEncoder setBuffer:_mX2      offset:0 atIndex:3 ];
            [ computeEncoder setBuffer:_mX1      offset:0 atIndex:4 ];
        }
        [ computeEncoder setBuffer:_mError   offset:0 atIndex:5 ];
        [ computeEncoder setBuffer:_mConst   offset:0 atIndex:6 ];

        int numGroupsPerGrid   = 1;
        int numThreadsPerGroup =   ( _mDim >= 1024 )
                                 ? 1024
                                 : ( ( (_mDim + 31) / 32) * 32 ) ;

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( numThreadsPerGroup, 1, 1) ];

        [computeEncoder endEncoding];

        [commandBuffer commit];

        [commandBuffer waitUntilCompleted];

        _mX1ToX2 = ! _mX1ToX2;
    }
}

@end

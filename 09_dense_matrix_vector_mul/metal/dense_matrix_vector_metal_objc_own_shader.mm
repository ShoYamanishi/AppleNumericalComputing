#include <memory>
#include <algorithm>
#include <iostream>

#import "dense_matrix_vector_metal_objc_own_shader.h"

@implementation DenseMatrixVectorMetalObjCOwnShader
{
    int _mColMajor;
    int _mThreadsOverRows;
    int _mM;
    int _mN;

    id<MTLDevice>               _mDevice;
    id<MTLComputePipelineState> _mPSO;

    id<MTLCommandQueue>         _mCommandQueue;

    id<MTLBuffer>               _mMat;
    id<MTLBuffer>               _mVec;
    id<MTLBuffer>               _mOutVec;
    id<MTLBuffer>               _mConst;
}

- (instancetype) initWithM:(int) m N:(int) n ColMajor:(bool) col_major ThreadsOverRows:(bool) threads_over_rows;
{
    self = [super init];
    if (self)
    {
        _mColMajor        = col_major;
        _mThreadsOverRows = threads_over_rows;
        _mM    = m;
        _mN    = n;

        [ self loadLibraryWithName: @"./dense_matrix_vector.metallib" ];

        if ( _mColMajor ) {
            if ( _mThreadsOverRows ) {
                _mPSO = [ self getPipelineStateForFunction: @"mult_col_major_threads_over_rows" ];
            }
            else {
                _mPSO = [ self getPipelineStateForFunction: @"mult_col_major_threads_over_columns" ];
            }
        }
        else {
            if ( _mThreadsOverRows ) {
                _mPSO = [ self getPipelineStateForFunction: @"mult_row_major_threads_over_rows" ];
                                                              
            }
            else {
                _mPSO = [ self getPipelineStateForFunction: @"mult_row_major_threads_over_columns" ];
            }
        }

        _mMat    = [ self getSharedMTLBufferForBytes: sizeof(float) * _mM * _mN for : @"_mMat"    ];
        _mVec    = [ self getSharedMTLBufferForBytes: sizeof(float) * _mN       for : @"_mVec"    ];
        _mOutVec = [ self getSharedMTLBufferForBytes: sizeof(float) * _mM       for : @"_mOutVec" ];
        _mConst  = [ self getSharedMTLBufferForBytes: sizeof(struct dense_matrix_vector_constants) for : @"_mConst" ];
        struct dense_matrix_vector_constants c;
        memset( &c, (uint)0, sizeof(struct dense_matrix_vector_constants) );
        c.M = _mM;
        c.N = _mN;
        memcpy( _mConst.contents, &c, sizeof(struct dense_matrix_vector_constants) );
    }
    return self;
}


- (void) setInitialStatesMat:(float*) mat Vec:(float*) v;
{
    memcpy( _mMat.contents, mat, _mM * _mN * sizeof(float) );
    memcpy( _mVec.contents, v,   _mN * sizeof(float) );
}


- (float*) getRawPointerOutVec
{
    return (float*)_mOutVec.contents;
}


- (float*) getRawPointerVec
{
    return (float*)_mVec.contents;
}

- (float*) getRawPointerMat
{
    return (float*)_mMat.contents;
}

- (void) performComputation
{
    memset( _mOutVec.contents, 0, sizeof(float) * _mM );

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO ];

    [ computeEncoder setBuffer:_mMat     offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mVec     offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mOutVec  offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst   offset:0 atIndex:3 ];

    int numGroupsPerGrid;
    int numThreadsPerGroup;

    if ( _mThreadsOverRows ) {

        numGroupsPerGrid   = (_mM + 1023) / 1024;
        numThreadsPerGroup =   ( _mM >= 1024 )
                             ? 1024
                             : ( ( (_mM + 31) / 32) * 32 ) ;
    }
    else {
        numGroupsPerGrid   = _mM;
        numThreadsPerGroup =   ( _mN >= 1024 )
                             ? 1024
                             : ( ( (_mN + 31) / 32) * 32 ) ;
    }
    [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( numThreadsPerGroup, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

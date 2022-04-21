#include <memory>
#include <iostream>

#import "dot_metal_objc.h"

struct dot_constants
{
    uint  num_elements;
};

@implementation DotMetalObjC
{
    int           _mReductionType;
    id<MTLComputePipelineState> _mPSOPass1;
    id<MTLComputePipelineState> _mPSOPass2;
    id<MTLComputePipelineState> _mPSO;

    id<MTLBuffer> _mX;
    id<MTLBuffer> _mY;
    id<MTLBuffer> _mZ;
    id<MTLBuffer> _mSPartial;
    id<MTLBuffer> _mSint;
    id<MTLBuffer> _mThreadGroupCounter;
    id<MTLBuffer> _mDot;
    id<MTLBuffer> _mConstPass1;
    id<MTLBuffer> _mConstPass2;
    id<MTLBuffer> _mConst;

    uint          _mNumElements;
    uint          _mNumThreadsPerGroup;
    uint          _mNumGroupsPerGrid;
}


- (instancetype) initWithNumElements:(size_t) num_elements 
                       ReductionType:(int)    reduction_type
                  NumThreadsPerGroup:(size_t) num_threads_per_group
                    NumGroupsPerGrid:(size_t) num_groups_per_grid
{
    self = [super init];
    if (self)
    {
        _mReductionType      = reduction_type;
        _mNumElements        = num_elements;
        _mNumThreadsPerGroup = num_threads_per_group;
        _mNumGroupsPerGrid   = num_groups_per_grid;

        [ self loadLibraryWithName:@"./dot.metallib" ];

        switch( reduction_type ) {

          case 1:
          case 2:
          case 3:
          case 4:
            [ self initType1234 ];
            break;

          case 5:
          case 6:
            [ self initType56 ];
            break;

          case 7:
            [ self initType7 ];
            break;

          default:
            break;
        }
    }
    return self;
}

-(void) initType1234
{
    if ( _mReductionType == 1 ) {
        _mPSOPass1 = [ self getPipelineStateForFunction:@"dot_type1_pass1" ];
        _mPSOPass2 = [ self getPipelineStateForFunction:@"dot_type1_pass2" ];
    }
    else if ( _mReductionType == 2 ) {
        _mPSOPass1 = [ self getPipelineStateForFunction:@"dot_type2_threadgroup_memory_pass1" ];
        _mPSOPass2 = [ self getPipelineStateForFunction:@"dot_type2_threadgroup_memory_pass2" ];
    }
    else if ( _mReductionType == 3 ) {
        _mPSOPass1 = [ self getPipelineStateForFunction:@"dot_type3_pass1_simd_shuffle" ];
        _mPSOPass2 = [ self getPipelineStateForFunction:@"dot_type3_pass2_simd_shuffle" ];
    }
    else {
        _mPSOPass1 = [ self getPipelineStateForFunction:@"dot_type4_pass1_simd_add" ];
        _mPSOPass2 = [ self getPipelineStateForFunction:@"dot_type4_pass2_simd_add" ];
    }

    _mX          = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumElements      for:@"_mX"   ];
    _mY          = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumElements      for:@"_mY"   ];
    _mZ          = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumGroupsPerGrid for:@"_mZ"   ];

    if ( _mReductionType == 1 ) {
        _mSPartial = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumElements for:@"_mSPartial"   ];
    }
    _mDot        = [ self getSharedMTLBufferForBytes: sizeof(float)                      for:@"_mDot" ];
    _mConstPass1 = [ self getSharedMTLBufferForBytes: sizeof(struct dot_constants)       for:@"_mConstPass1" ];
    _mConstPass2 = [ self getSharedMTLBufferForBytes: sizeof(struct dot_constants)       for:@"_mConstPass2" ];

    struct dot_constants c;
    memset( &c, (int)0, sizeof(struct dot_constants) );
    c.num_elements = _mNumElements;
    memcpy( _mConstPass1.contents, &c, sizeof(struct dot_constants) );
    c.num_elements = _mNumGroupsPerGrid;
    memcpy( _mConstPass2.contents, &c, sizeof(struct dot_constants) );
}

-(void) initType56
{
    if ( _mReductionType == 5) {
        _mPSO = [ self getPipelineStateForFunction:@"dot_type5_atomic_simd_shuffle" ];
    }
    else {
        _mPSO = [ self getPipelineStateForFunction:@"dot_type6_atomic_simd_add" ];
    }

    _mX     = [ self getSharedMTLBufferForBytes:  sizeof(float) * _mNumElements for:@"_mX"     ];
    _mY     = [ self getSharedMTLBufferForBytes:  sizeof(float) * _mNumElements for:@"_mY"     ];
    _mSint  = [ self getSharedMTLBufferForBytes: sizeof(uint)                  for:@"_mSint"  ];
    _mConst = [ self getSharedMTLBufferForBytes:  sizeof(struct dot_constants)  for:@"_mConst" ];

    struct dot_constants c;
    memset( &c, (int)0, sizeof(struct dot_constants) );
    c.num_elements = _mNumElements;
    memcpy( _mConst.contents, &c, sizeof(struct dot_constants) );
}

-(void) initType7
{
    _mPSO = [ self getPipelineStateForFunction:@"dot_type7_atomic_thread_group_counter" ];

    _mX     = [ self getSharedMTLBufferForBytes:  sizeof(float) * _mNumElements      for:@"_mX"     ];
    _mY     = [ self getSharedMTLBufferForBytes:  sizeof(float) * _mNumElements      for:@"_mY"     ];
    _mZ     = [ self getSharedMTLBufferForBytes:  sizeof(float) * _mNumGroupsPerGrid for:@"_mZ"     ];
    _mDot   = [ self getSharedMTLBufferForBytes:  sizeof(float)                      for:@"_mDot"   ];
    _mThreadGroupCounter
            = [ self getSharedMTLBufferForBytes:  sizeof(uint)                       for:@"_mThreadGroupCounter"   ];
    _mSint  = [ self getSharedMTLBufferForBytes: sizeof(uint)                       for:@"_mSint"  ];
    _mConst = [ self getSharedMTLBufferForBytes:  sizeof(struct dot_constants)       for:@"_mConst" ];

    struct dot_constants c;
    memset( &c, (int)0, sizeof(struct dot_constants) );
    c.num_elements = _mNumElements;
    memcpy( _mConst.contents, &c, sizeof(struct dot_constants) );
}

-(float*) getRawPointerX
{
    return (float*)_mX.contents;
}

-(float*) getRawPointerY
{
    return (float*)_mY.contents;
}

- (float)  getDotXY
{
    switch( _mReductionType ) {

      case 1:
      case 2:
      case 3:
      case 4:
        return ((float*)_mDot.contents)[0];
        break;

      case 5:
      case 6:
        return ((float*)_mSint.contents)[0];

      case 7:
        return ((float*)_mDot.contents)[0];
        break;

      default:
        break;
    }
    return 0.0;
}

-(void) performComputation
{
    switch( _mReductionType ) {

      case 1:
      case 2:
      case 3:
      case 4:
        [ self performComputationType1234 ];
        break;

      case 5:
      case 6:
        [ self performComputationType56 ];
        break;

      case 7:
        [ self performComputationType7 ];
        break;

      default:
        break;
    }
}

-(void) performComputationType1234
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState:_mPSOPass1 ];

    [ computeEncoder setBuffer:_mX          offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mY          offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mZ          offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConstPass1 offset:0 atIndex:3 ];

    if ( _mReductionType  ==  1 ) {
        [ computeEncoder setBuffer:_mSPartial offset:0 atIndex:4 ];
    }
    else {
        [ computeEncoder setThreadgroupMemoryLength:sizeof(float)*_mNumThreadsPerGroup atIndex:0 ];
    }

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1 )
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ computeEncoder setComputePipelineState:_mPSOPass2 ];

    [ computeEncoder setBuffer:_mZ          offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mDot        offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mConstPass2 offset:0 atIndex:2 ];

    if ( _mReductionType  ==  1 ) {
        [ computeEncoder setBuffer:_mSPartial offset:0 atIndex:3 ];
    }
    else {
        [ computeEncoder setThreadgroupMemoryLength:sizeof(float)*_mNumThreadsPerGroup atIndex:0 ];
    }

    [ computeEncoder dispatchThreadgroups:MTLSizeMake(                    1, 1, 1 )
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

-(void) performComputationType56
{
    memset( _mSint.contents, 0, sizeof(uint) );

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState:_mPSO ];

    [ computeEncoder setBuffer:_mX          offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mY          offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mSint       offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst      offset:0 atIndex:3 ];

    [ computeEncoder setThreadgroupMemoryLength:sizeof(float)*_mNumThreadsPerGroup atIndex:0 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

-(void) performComputationType7
{
    memset( _mThreadGroupCounter.contents, 0, sizeof(uint)                       );
    memset( _mZ.contents,                  0, _mNumGroupsPerGrid * sizeof(float) );

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState:_mPSO ];

    [ computeEncoder setBuffer:_mX          offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mY          offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mZ          offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mDot        offset:0 atIndex:3 ];
    [ computeEncoder setBuffer:_mThreadGroupCounter 
                                            offset:0 atIndex:4 ];
    [ computeEncoder setBuffer:_mConst      offset:0 atIndex:5 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

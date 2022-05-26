#include <iostream>
#include <memory>
#include <algorithm>

#import "prefix_sum_metal_objc_merrill_grimshaw.h"

#define NUM_THREADS_PER_THREADGROUP 1024

struct prefix_sum_constants
{
    uint  num_elements;
    uint  num_threads_per_partial_sum;
};

inline uint roundup_X(uint x, uint n )
{
    return ((n + x - 1) / x) * x;
}

inline uint roundup_1024(uint n )
{
    return ((n + 1023) / 1024) * 1024;
}

inline uint roundup_32(uint n)
{
    return ((n + 31) / 32) * 32;
}

inline uint roundup_16(uint n)
{
    return ((n + 15) / 16) * 16;
}

@implementation PrefixSumMetalObjCMerrillGrimshaw
{

    id<MTLBuffer> _mIn;
    id<MTLBuffer> _mOut;

    int           _mNumThreadsPerThreadgroup;

    uint          _mNumPartialSumsRequested;
    uint          _mNumPartialSums;
    uint          _mNumThreadsPerPartialSum;

    id<MTLComputePipelineState> _mPSO_get_partial_sums_32_X;
    id<MTLComputePipelineState> _mPSO_scan_threadgroupwise_32_X;
    id<MTLComputePipelineState> _mPSO_scan_final_32_X;

    id<MTLBuffer> _mPartialSums;

    uint          _mNumElementsStep1;
    uint          _mNumThreadsPerGroupStep1;
    uint          _mNumGroupsPerGridStep1;
    id<MTLBuffer> _mConstStep1;

    uint          _mNumElementsStep2;
    uint          _mNumThreadsPerGroupStep2;
    uint          _mNumGroupsPerGridStep2;
    id<MTLBuffer> _mConstStep2;

    uint          _mNumElementsStep3;
    uint          _mNumThreadsPerGroupStep3;
    uint          _mNumGroupsPerGridStep3;
    id<MTLBuffer> _mConstStep3;
}

- (instancetype) initWithNumElements:(size_t) num_elements
                      NumPartialSums:(size_t) num_partial_sums 
                            ForFloat:(bool)   for_float
            NumThreadsPerThreadgroup:(int)    num_threads_per_threadgroup
{
    self = [super init];
    if (self)
    {
        _mNumThreadsPerThreadgroup = num_threads_per_threadgroup;

        [ self loadLibraryWithName:@"./prefix_sum.metallib" ];

        [ self findConfiguration: num_elements requestedNumPartialSums:num_partial_sums ];

        if ( for_float )  {
            _mPSO_get_partial_sums_32_X     = [ self getPipelineStateForFunction: @"mg_get_partial_sums_32_X_float"     ];
            _mPSO_scan_threadgroupwise_32_X = [ self getPipelineStateForFunction: @"mg_scan_threadgroupwise_32_X_float" ];
            _mPSO_scan_final_32_X           = [ self getPipelineStateForFunction: @"mg_scan_final_32_X_float"           ];
        }
        else {
            _mPSO_get_partial_sums_32_X     = [ self getPipelineStateForFunction: @"mg_get_partial_sums_32_X_int"     ];
            _mPSO_scan_threadgroupwise_32_X = [ self getPipelineStateForFunction: @"mg_scan_threadgroupwise_32_X_int" ];
            _mPSO_scan_final_32_X           = [ self getPipelineStateForFunction: @"mg_scan_final_32_X_int"           ];
        }

        _mIn          = [ self getSharedMTLBufferForBytes:  _mNumElementsStep1 * (for_float?sizeof(float):sizeof(int)) for:@"_mIn"          ];
        _mOut         = [ self getSharedMTLBufferForBytes:  _mNumElementsStep1 * (for_float?sizeof(float):sizeof(int)) for:@"_mOut"         ];
        _mPartialSums = [ self getPrivateMTLBufferForBytes: _mNumPartialSums   * (for_float?sizeof(float):sizeof(int)) for:@"_mPartialSums" ];

        _mConstStep1  = [ self getSharedMTLBufferForBytes:  sizeof(struct prefix_sum_constants) for:@"_mConstStep1" ];
        struct prefix_sum_constants c;
        memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );
        c.num_elements                = _mNumElementsStep1;
        c.num_threads_per_partial_sum = _mNumThreadsPerPartialSum;
        memcpy( _mConstStep1.contents, &c, sizeof(struct prefix_sum_constants) );

        _mConstStep2  = [ self getSharedMTLBufferForBytes:  sizeof(struct prefix_sum_constants) for:@"_mConstStep2" ];
        memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );
        c.num_elements                = _mNumElementsStep2;
        c.num_threads_per_partial_sum = 0;
        memcpy( _mConstStep2.contents, &c, sizeof(struct prefix_sum_constants) );

        _mConstStep3  = [ self getSharedMTLBufferForBytes:  sizeof(struct prefix_sum_constants) for:@"_mConstStep3" ];
        memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );
        c.num_elements                = _mNumElementsStep3;
        c.num_threads_per_partial_sum = _mNumThreadsPerPartialSum;
        memcpy( _mConstStep3.contents, &c, sizeof(struct prefix_sum_constants) );
    }
    return self;
}

-(void) findConfiguration:(uint)   num_elements
  requestedNumPartialSums:(size_t) num_partial_sums_requested
{

    _mNumPartialSumsRequested = std::min( (uint)NUM_THREADS_PER_THREADGROUP, (uint)num_partial_sums_requested );

    _mNumThreadsPerPartialSum = roundup_X( 
        _mNumThreadsPerThreadgroup,
        ( num_elements + num_partial_sums_requested - 1 ) / num_partial_sums_requested 
    );

    _mNumPartialSums = (num_elements + _mNumThreadsPerPartialSum - 1) / _mNumThreadsPerPartialSum;

    _mNumElementsStep1        = num_elements;
    _mNumThreadsPerGroupStep1 = _mNumThreadsPerThreadgroup;
    _mNumGroupsPerGridStep1   = _mNumPartialSums;

    _mNumElementsStep2        = _mNumPartialSums;
    _mNumThreadsPerGroupStep2 = roundup_32( _mNumPartialSums );
    _mNumGroupsPerGridStep2   = 1;

    _mNumElementsStep3        = num_elements;;
    _mNumThreadsPerGroupStep3 = _mNumThreadsPerThreadgroup;
    _mNumGroupsPerGridStep3   = _mNumPartialSums;

//    std::cerr << "_mNumElementsStep1: "        << _mNumElementsStep1 << "\n";
//    std::cerr << "_mNumThreadsPerGroupStep1: " << _mNumThreadsPerGroupStep1 << "\n";
//    std::cerr << "_mNumGroupsPerGridStep1: "   << _mNumGroupsPerGridStep1 << "\n";

//    std::cerr << "_mNumElementsStep2: "        << _mNumElementsStep2 << "\n";
//    std::cerr << "_mNumThreadsPerGroupStep2: " << _mNumThreadsPerGroupStep2 << "\n";
//    std::cerr << "_mNumGroupsPerGridStep2: "   << _mNumGroupsPerGridStep2 << "\n";

//    std::cerr << "_mNumElementsStep3: "        << _mNumElementsStep3 << "\n";
//    std::cerr << "_mNumThreadsPerGroupStep3: " << _mNumThreadsPerGroupStep3 << "\n";
//    std::cerr << "_mNumGroupsPerGridStep3: "   << _mNumGroupsPerGridStep3 << "\n";
}

- (uint) numElements
{
    return _mNumElementsStep1;
}

-(int*) getRawPointerInForInt
{
    return (int*)_mIn.contents;
}


-(float*) getRawPointerInForFloat
{
    return (float*)_mIn.contents;
}

-(int*) getRawPointerOutForInt
{
    return (int*)_mOut.contents;
}

-(float*) getRawPointerOutForFloat
{
    return (float*)_mOut.contents;
}

-(int*) getRawPointerPartialSumsForInt
{
    return (int*)_mPartialSums.contents;
}

-(float*) getRawPointerPartialSumsForFloat
{
    return (float*)_mPartialSums.contents;
}


-(void) performComputation
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO_get_partial_sums_32_X ];

    [ computeEncoder setBuffer:_mIn          offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mPartialSums offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mConstStep1  offset:0 atIndex:2 ];
    [ computeEncoder setThreadgroupMemoryLength: roundup_16(sizeof(int)*_mNumThreadsPerGroupStep1 / 32) atIndex:0 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridStep1,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupStep1, 1, 1) ];

    [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_32_X ];

    [ computeEncoder setBuffer:_mPartialSums offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mConstStep2  offset:0 atIndex:1 ];
    [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupStep2 / 32) atIndex:0 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridStep2,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupStep2, 1, 1) ];

    [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ computeEncoder setComputePipelineState: _mPSO_scan_final_32_X ];

    [ computeEncoder setBuffer:_mIn          offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mOut         offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mPartialSums offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConstStep3  offset:0 atIndex:3 ];

    [ computeEncoder setThreadgroupMemoryLength:sizeof(int)*_mNumThreadsPerGroupStep3 / 32 atIndex:0 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridStep3,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupStep3, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

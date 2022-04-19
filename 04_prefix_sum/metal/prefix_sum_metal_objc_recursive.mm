#include <memory>
#include <algorithm>

#import "prefix_sum_metal_objc_recursive.h"

struct prefix_sum_constants
{
    uint  num_elements;
};

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

// Type 1: scan-then-fan
//
//   - 0 < num_elements <= 1024
//     scan_threadgroupwise_intermediate_32_32
//
//   - 1024 < num_elements <= 1024*1024
//     scan_threadgroupwise_intermediate_32_32
//     scan_threadgroupwise_intermediate_32_32
//     add_base_32_32
//
//   - 1024*1024 < num_elements <= 1024*1024*1024
//     scan_threadgroupwise_intermediate_32_32
//     scan_threadgroupwise_intermediate_32_32
//     scan_threadgroupwise_intermediate_32_32
//     add_base_32_32
//     add_base_32_32
//
// Type 2: reduce-then_scan
//
//   - 0 < num_elements <= 1024
//     scan_threadgroupwise_intermediate_32_32
//
//   - 1024 < num_elements <= 1024*1024
//     sum_threadgroup_32_32
//     scan_threadgroupwise_intermediate_32_32
//     scan_with_base_threadgroupwise_32_32
//
//   - 1024*1024 < num_elements <= 1024*1024*1024
//     sum_threadgroup_32_32
//     sum_threadgroup_32_32
//     scan_threadgroupwise_intermediate_32_32
//     scan_with_base_threadgroupwise_32_32
//     scan_with_base_threadgroupwise_32_32

@implementation PrefixSumMetalObjCRecursive
{
    int _mAlgoType;

    // scan-then-fan & reduce-then-scan
    id<MTLComputePipelineState> _mPSO_scan_threadgroupwise_intermediate_32_32;
    id<MTLComputePipelineState> _mPSO_add_base_32_32;
    id<MTLComputePipelineState> _mPSO_sum_threadgroup_32_32;
    id<MTLComputePipelineState> _mPSO_scan_with_base_threadgroupwise_32_32;

    id<MTLBuffer> _mIn;
    id<MTLBuffer> _mOut;

    id<MTLBuffer> _mConstLayer1;
    id<MTLBuffer> _mConstLayer2;
    id<MTLBuffer> _mConstLayer3;

    uint          _mConfiguration;

    // Layer 1
    uint          _mNumElementsLayer1;
    uint          _mNumThreadsPerGroupLayer1;
    uint          _mNumGroupsPerGridLayer1;
    id<MTLBuffer> _mGridPrefixSumsLayer1;

    // Layer 2
    uint          _mNumElementsLayer2;
    uint          _mNumThreadsPerGroupLayer2;
    uint          _mNumGroupsPerGridLayer2;
    id<MTLBuffer> _mGridPrefixSumsLayer2;

    // Layer 3
    uint          _mNumElementsLayer3;
    uint          _mNumThreadsPerGroupLayer3;
    uint          _mNumGroupsPerGridLayer3;
    id<MTLBuffer> _mGridPrefixSumsLayer3;

}

- (instancetype) initWithNumElements:(size_t) num_elements
                                Type:(int)    algo_type
                      NumPartialSums:(size_t) num_partial_sums
                            ForFloat:(bool)   for_float
{
    self = [super init];
    if (self)
    {
        [ self loadLibraryWithName:@"./prefix_sum.metallib" ];

        _mAlgoType = algo_type;

        [self findConfiguration:num_elements ];

        if ( algo_type == 1 || algo_type == 2 ) {

            [ self findConfiguration:num_elements ];

            if (for_float) {
                _mPSO_scan_threadgroupwise_intermediate_32_32 = [ self getPipelineStateForFunction: @"scan_threadgroupwise_intermediate_32_32_float" ];
                _mPSO_add_base_32_32                          = [ self getPipelineStateForFunction: @"add_base_32_32_float"                          ];
                _mPSO_sum_threadgroup_32_32                   = [ self getPipelineStateForFunction: @"sum_threadgroup_32_32_float"                   ];
                _mPSO_scan_with_base_threadgroupwise_32_32    = [ self getPipelineStateForFunction: @"scan_with_base_threadgroupwise_32_32_float"    ];
            }
            else {
                _mPSO_scan_threadgroupwise_intermediate_32_32 = [ self getPipelineStateForFunction: @"scan_threadgroupwise_intermediate_32_32_int" ];
                _mPSO_add_base_32_32                          = [ self getPipelineStateForFunction: @"add_base_32_32_int"                          ];
                _mPSO_sum_threadgroup_32_32                   = [ self getPipelineStateForFunction: @"sum_threadgroup_32_32_int"                   ];
                _mPSO_scan_with_base_threadgroupwise_32_32    = [ self getPipelineStateForFunction: @"scan_with_base_threadgroupwise_32_32_int"    ];
            }

            _mIn  = [ self getSharedMTLBufferForBytes: _mNumElementsLayer1 * (for_float?sizeof(float):sizeof(int)) for:@"_mIn"   ];
            _mOut = [ self getSharedMTLBufferForBytes: _mNumElementsLayer1 * (for_float?sizeof(float):sizeof(int)) for:@"_mOut"  ];

            if ( _mConfiguration == 1 || _mConfiguration == 2 ||_mConfiguration == 3 ) {

                _mGridPrefixSumsLayer1 = [ self getPrivateMTLBufferForBytes: _mNumGroupsPerGridLayer1 * (for_float?sizeof(float):sizeof(int))
                                                                        for: @"_mGridPrefixSumsLayer1" ];
                _mConstLayer1          = [ self getSharedMTLBufferForBytes:  sizeof(struct prefix_sum_constants)    for:@"_mConstLayer1"          ];

                struct prefix_sum_constants c;
                memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );
                c.num_elements = _mNumElementsLayer1;
                memcpy( _mConstLayer1.contents, &c, sizeof(struct prefix_sum_constants) );
            }

            if ( _mConfiguration == 2 ||_mConfiguration == 3 ) {

                _mGridPrefixSumsLayer2 = [ self getPrivateMTLBufferForBytes: _mNumGroupsPerGridLayer2 * (for_float?sizeof(float):sizeof(int))
                                                                        for: @"_mGridPrefixSumsLayer2"  ];
                _mConstLayer2          = [ self getSharedMTLBufferForBytes:  sizeof(struct prefix_sum_constants)    for:@"_mConstLayer2"  ];

                struct prefix_sum_constants c;
                memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );
                c.num_elements = _mNumElementsLayer2;
                memcpy( _mConstLayer2.contents, &c, sizeof(struct prefix_sum_constants) );
            }

            if ( _mConfiguration == 3 ) {

                _mGridPrefixSumsLayer3 = [ self getPrivateMTLBufferForBytes: _mNumGroupsPerGridLayer3 * (for_float?sizeof(float):sizeof(int))
                                                                        for: @"_mGridPrefixSumsLayer3"  ];
                _mConstLayer3          = [ self getSharedMTLBufferForBytes:  sizeof(struct prefix_sum_constants)    for:@"_mConstLayer3"  ];

                struct prefix_sum_constants c;
                memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );
                c.num_elements = _mNumElementsLayer3;
                memcpy( _mConstLayer3.contents, &c, sizeof(struct prefix_sum_constants) );
            }
        }
    }
    return self;
}


-(void) findConfiguration:(uint) num_elements
{
    if ( num_elements <= 1024 ) {

        _mConfiguration = 1;

        _mNumElementsLayer1        = num_elements;
        _mNumElementsLayer2        = 0;
        _mNumElementsLayer3        = 0;

        _mNumThreadsPerGroupLayer1 = roundup_32( _mNumElementsLayer1 );
        _mNumGroupsPerGridLayer1   = 1;

        _mNumThreadsPerGroupLayer2 = 0;
        _mNumGroupsPerGridLayer2   = 0;

        _mNumThreadsPerGroupLayer3 = 0;
        _mNumGroupsPerGridLayer3   = 0;
    }
    else if  ( num_elements <= 1024*1024 ) {

        _mConfiguration = 2;

        _mNumElementsLayer1        = num_elements;
        _mNumElementsLayer2        = roundup_1024( _mNumElementsLayer1 ) / 1024;
        _mNumElementsLayer3        = 0;

        _mNumThreadsPerGroupLayer1 = 1024;
        _mNumGroupsPerGridLayer1   = roundup_32( _mNumElementsLayer2 );

        _mNumThreadsPerGroupLayer2 = _mNumGroupsPerGridLayer1;
        _mNumGroupsPerGridLayer2   = 1;

        _mNumThreadsPerGroupLayer3 = 0;
        _mNumGroupsPerGridLayer3   = 0;

    }
    else if  ( num_elements <= 1024*1024*1024 ) {

        _mConfiguration = 3;

        _mNumElementsLayer1        = num_elements;
        _mNumElementsLayer2        = roundup_1024( _mNumElementsLayer1 ) / 1024;
        _mNumElementsLayer3        = roundup_1024( _mNumElementsLayer2 ) / 1024;

        _mNumThreadsPerGroupLayer1 = 1024;
        _mNumGroupsPerGridLayer1   = roundup_32( _mNumElementsLayer2 );

        _mNumThreadsPerGroupLayer2 = 1024;
        _mNumGroupsPerGridLayer2   = roundup_32( _mNumElementsLayer3 );

        _mNumThreadsPerGroupLayer3 = _mNumGroupsPerGridLayer2;
        _mNumGroupsPerGridLayer3   = 1;
    }
    else {

        _mConfiguration = 0;

        _mNumElementsLayer2        = 0;
        _mNumElementsLayer3        = 0;

        _mNumThreadsPerGroupLayer1 = 0;
        _mNumGroupsPerGridLayer1   = 0;

        _mNumThreadsPerGroupLayer2 = 0;
        _mNumGroupsPerGridLayer2   = 0;

        _mNumThreadsPerGroupLayer3 = 0;
        _mNumGroupsPerGridLayer3   = 0;
    }
}


- (uint) numThreadsPerGroup:(uint) layer
{
    switch (layer) {
      case 1:
        return _mNumThreadsPerGroupLayer1;
      case 2:
        return _mNumThreadsPerGroupLayer2;
      case 3:
        return _mNumThreadsPerGroupLayer3;
      default:
        return 0;
    }
}

- (uint) numGroupsPerGrid:(uint) layer
{
    switch (layer) {
      case 1:
          return _mNumGroupsPerGridLayer1;
      case 2:
          return _mNumGroupsPerGridLayer2;
      case 3:
          return _mNumGroupsPerGridLayer3;
      default:
        return 0;
    }
}


- (uint) numElements:(uint) layer
{
    switch (layer) {
      case 1:
          return _mNumElementsLayer1;
      case 2:
          return _mNumElementsLayer2;
      case 3:
          return _mNumElementsLayer3;
      default:
        return 0;
    }
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


-(int*) getRawPointerGridPrefixSumsForInt:(uint)layer;
{
    switch (layer) {
      case 1:
          return (int*)_mGridPrefixSumsLayer1.contents;
      case 2:
          return (int*)_mGridPrefixSumsLayer2.contents;
      case 3:
          return (int*)_mGridPrefixSumsLayer2.contents;
      default:
        return nullptr;
    }
}


-(float*) getRawPointerGridPrefixSumsForFloat:(uint)layer;
{
    switch (layer) {
      case 1:
          return (float*)_mGridPrefixSumsLayer1.contents;
      case 2:
          return (float*)_mGridPrefixSumsLayer2.contents;
      case 3:
          return (float*)_mGridPrefixSumsLayer2.contents;
      default:
        return nullptr;
    }
}



- (void) performComputation
{
    if ( _mAlgoType == 1 ) {
        [ self performComputationScanThenFan ];
    }
    else {
        [ self performComputationReduceThenScan ];
    }
}


-(void) performComputationScanThenFan
{

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    if (_mConfiguration == 1) {

        [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];

    }
    else if (_mConfiguration == 2) {

        [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:3 ];

        [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer2           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupLayer2) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer2,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer2, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setComputePipelineState: _mPSO_add_base_32_32 ];

        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:2 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];
    }

    else if (_mConfiguration == 3) {

        [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:3 ];

        [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer2           offset:0 atIndex:3 ];

        [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupLayer2) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer2,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer2, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer3  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer3           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_16(sizeof(int)*_mNumThreadsPerGroupLayer3) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer3,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer3, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setComputePipelineState: _mPSO_add_base_32_32 ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mConstLayer2           offset:0 atIndex:2 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer2,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer2, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:2 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];
    }

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}



-(void) performComputationReduceThenScan
{

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    if (_mConfiguration == 1) {

        [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ computeEncoder setBuffer:_mIn                     offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mOut                    offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1   offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer1            offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];

    }
    else if (_mConfiguration == 2) {

        [ computeEncoder setComputePipelineState: _mPSO_sum_threadgroup_32_32 ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:2 ];

        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer2           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer2) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer2,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer2, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_32 ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];
    }

    else if (_mConfiguration == 3) {


        [ computeEncoder setComputePipelineState: _mPSO_sum_threadgroup_32_32 ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:2 ];

        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mConstLayer2           offset:0 atIndex:2 ];

        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer2) atIndex:0 ];

        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer2,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer2, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer3  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer3           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer3) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer3,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer3, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_32 ];

        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer2  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer2           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer2) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer2,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer2, 1, 1) ];

        [ computeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
        [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:1 ];
        [ computeEncoder setBuffer:_mGridPrefixSumsLayer1  offset:0 atIndex:2 ];
        [ computeEncoder setBuffer:_mConstLayer1           offset:0 atIndex:3 ];
        [ computeEncoder setThreadgroupMemoryLength:roundup_32(sizeof(int)*_mNumThreadsPerGroupLayer1) atIndex:0 ];
        [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGridLayer1,   1, 1)
                        threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroupLayer1, 1, 1) ];
    }

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

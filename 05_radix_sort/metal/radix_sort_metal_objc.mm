#include <memory>
#include <algorithm>

#import "radix_sort_metal_objc.h"

struct radix_sort_constants
{
    uint  total_num_elements;
    uint  bit_right_shift;
    bool  flip_msb;
    bool  for_float;
};

struct prefix_sum_constants
{
    uint  num_elements;
};

struct parallel_order_checking_constants
{
    uint  total_num_elements;
};

// Strategy:
//
// for each shift in (0, 32, step 2)
// do:
//     % pre-check and early out
//
//     is_sorted_within_threadgroups()
//     are_all_less_than_equal()
//
//     if sorted:
//     then
//         break
//     endif
//
//     % radix sort for 2-consecutive bits within threadgroup
// 
//     four_way_prefix_sum_with_inblock_shuffle()
//
//     % prefix-sum of theradgroups
// 
//     else if num elements <= 1024:
//     then
//         scan_threadgroupwise_intermediate_32_32()
// 
//     else if 1024 < num elements <= 1024*1024:
//     then
//         sum_threadgroup_32_32()
//         scan_threadgroupwise_intermediate_32_32()
//         scan_with_base_threadgroupwise_32_32()
//
//     else if 1024*1024 < num elements <= 1024*1024*1024:
//     then
//         sum_threadgroup_32_32()
//         sum_threadgroup_32_32()
//         scan_threadgroupwise_intermediate_32_32()
//         scan_with_base_threadgroupwise_32_32()
//         scan_with_base_threadgroupwise_32_32()
//     endif
// 
//     % final sort In1 => In2 (even) or In2 => In1 (odd)
//     coalesced_block_mapping_for_the_n_chunk_input()
// done:


static inline uint alignUpAndDivide(uint v, uint d )
{
    return (v + d - 1) / d;
}


static inline uint alignUp(uint v, uint a )
{
    return ( (v + a - 1) / a ) * a;
}


class PrefixSumLayerParams {

  public:

    PrefixSumLayerParams()

        :num_elements_               (0)
        ,num_threads_per_threadgroup_(0)
        ,num_threadgroups_per_grid_  (0){;}

    void set(

        const uint num_elements,
        const uint num_threads_per_threadgroup,
        const uint num_threadgroups_per_grid
    ) {
        num_elements_                = num_elements;
        num_threads_per_threadgroup_ = num_threads_per_threadgroup;
        num_threadgroups_per_grid_   = num_threadgroups_per_grid;
    }

    uint          num_elements_;
    uint          num_threads_per_threadgroup_;
    uint          num_threadgroups_per_grid_;
};


@implementation RadixSortMetalObjC {

    bool                        _mResultOn1;
    uint                        _mNumElements;
    uint                        _mNumThreadgroups;
    bool                        _mForFloat;
    bool                        _mCoalescedWrite;
    bool                        _mEarlyOut;

    uint                        _mPrefixSumConfiguration;

    PrefixSumLayerParams        _mPrefixSumParamsLayer1;
    PrefixSumLayerParams        _mPrefixSumParamsLayer2;
    PrefixSumLayerParams        _mPrefixSumParamsLayer3;

    id<MTLComputePipelineState> _mPSO_four_way_prefix_sum_with_inblock_shuffle;
    id<MTLComputePipelineState> _mPSO_coalesced_block_mapping_for_the_n_chunk_input;
    id<MTLComputePipelineState> _mPSO_uncoalesced_block_mapping_for_the_n_chunk_input;

    id<MTLComputePipelineState> _mPSO_is_sorted_within_threadgroups;
    id<MTLComputePipelineState> _mPSO_are_all_less_than_equal;

    id<MTLComputePipelineState> _mPSO_scan_threadgroupwise_intermediate_32_32;
    id<MTLComputePipelineState> _mPSO_sum_threadgroup_32_32;
    id<MTLComputePipelineState> _mPSO_scan_with_base_threadgroupwise_32_32;

    id<MTLBuffer> _mMetalArray1;
    id<MTLBuffer> _mMetalArray2;
    id<MTLBuffer> _mMetalThreadgroupBoundariesPrevLast;
    id<MTLBuffer> _mMetalThreadgroupBoundariesFirst;
    id<MTLBuffer> _mMetalArrayIsUnsorted;

    id<MTLBuffer> _mMetalConstRadixSortConstants;
    id<MTLBuffer> _mMetalConstParallelOrderCheckingConstants;

    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane0;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane1;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane2;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane3;

    id<MTLBuffer> _mMetalStartPosWithinThreadgroupLane1;
    id<MTLBuffer> _mMetalStartPosWithinThreadgroupLane2;
    id<MTLBuffer> _mMetalStartPosWithinThreadgroupLane3;

    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer1;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer2;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer3;

    id<MTLBuffer> _mMetalPrefixSumConstantsLayer1;
    id<MTLBuffer> _mMetalPrefixSumConstantsLayer2;
    id<MTLBuffer> _mMetalPrefixSumConstantsLayer3;
}


-(void) prefixSumFindConfiguration
{

    const uint num_elems_layer1 = alignUpAndDivide( _mNumElements,    1024 );
    const uint num_elems_layer2 = alignUpAndDivide( num_elems_layer1, 1024 );
    const uint num_elems_layer3 = alignUpAndDivide( num_elems_layer2, 1024 );

    if ( num_elems_layer1 <= 1024 ) {

        _mPrefixSumConfiguration = 1;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, alignUp( num_elems_layer1, 32 ), 1 );
    }
    else if  ( num_elems_layer1 <= 1024*1024 ) {

        _mPrefixSumConfiguration = 2;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, 1024, num_elems_layer2 );
        _mPrefixSumParamsLayer2.set( num_elems_layer2, alignUp( num_elems_layer2, 32 ), 1 );
    }
    else if  ( num_elems_layer1 <= 1024*1024*1024 ) {

        _mPrefixSumConfiguration = 3;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, 1024, num_elems_layer2 );
        _mPrefixSumParamsLayer1.set( num_elems_layer2, 1024, num_elems_layer3 );
        _mPrefixSumParamsLayer2.set( num_elems_layer3, alignUp( num_elems_layer3, 32 ), 1 );
    }
    else {
        _mPrefixSumConfiguration = 0;
    }
}


- (void) allocateMetalBuffers
{
    _mMetalArray1                              = [ self getSharedMTLBufferForBytes:  sizeof(int)*_mNumElements                for:@"_mMetalArray1"                              ];
    _mMetalArray2                              = [ self getSharedMTLBufferForBytes:  sizeof(int)*_mNumElements                for:@"_mMetalArray2"                              ];
    _mMetalThreadgroupBoundariesPrevLast       = [ self getSharedMTLBufferForBytes:  sizeof(int)*_mNumThreadgroups            for:@"_mMetalThreadgroupBoundariesPrevLast"       ];
    _mMetalThreadgroupBoundariesFirst          = [ self getSharedMTLBufferForBytes:  sizeof(int)*_mNumThreadgroups            for:@"_mMetalThreadgroupBoundariesFirst"          ];
    _mMetalArrayIsUnsorted                     = [ self getSharedMTLBufferForBytes:  sizeof(int)                              for:@"_mMetalArrayIsUnsorted"                     ];
    _mMetalConstRadixSortConstants             = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants"             ];
    _mMetalConstParallelOrderCheckingConstants = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstParallelOrderCheckingConstants" ];
    _mMetalPartialSumsPerThreadgroupLane0      = [ self getPrivateMTLBufferForBytes: sizeof(int)* _mNumThreadgroups           for:@"_mMetalPartialSumsPerThreadgroupLane0"      ];
    _mMetalPartialSumsPerThreadgroupLane1      = [ self getPrivateMTLBufferForBytes: sizeof(int)* _mNumThreadgroups           for:@"_mMetalPartialSumsPerThreadgroupLane1"      ];
    _mMetalPartialSumsPerThreadgroupLane2      = [ self getPrivateMTLBufferForBytes: sizeof(int)* _mNumThreadgroups           for:@"_mMetalPartialSumsPerThreadgroupLane2"      ];
    _mMetalPartialSumsPerThreadgroupLane3      = [ self getPrivateMTLBufferForBytes: sizeof(int)* _mNumThreadgroups           for:@"_mMetalPartialSumsPerThreadgroupLane3"      ];
    _mMetalStartPosWithinThreadgroupLane1      = [ self getPrivateMTLBufferForBytes: sizeof(unsigned short)*_mNumThreadgroups for:@"_mMetalStartPosWithinThreadgroupLane1"      ];
    _mMetalStartPosWithinThreadgroupLane2      = [ self getPrivateMTLBufferForBytes: sizeof(unsigned short)*_mNumThreadgroups for:@"_mMetalStartPosWithinThreadgroupLane2"      ];
    _mMetalStartPosWithinThreadgroupLane3      = [ self getPrivateMTLBufferForBytes: sizeof(unsigned short)*_mNumThreadgroups for:@"_mMetalStartPosWithinThreadgroupLane3"      ];

    if ( _mPrefixSumConfiguration == 1 || _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 ) {

        _mMetalPrefixSumGridPrefixSumLayer1    = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer1.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer1" ];
        _mMetalPrefixSumConstantsLayer1        = [ self getPrivateMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer1"     ];
    }
    if ( _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 ) {

        _mMetalPrefixSumGridPrefixSumLayer2    = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer2.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer2" ];
        _mMetalPrefixSumConstantsLayer2        = [ self getPrivateMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer2"     ];
    }
    if ( _mPrefixSumConfiguration == 3 ) {

        _mMetalPrefixSumGridPrefixSumLayer3    = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer3.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer3" ];
        _mMetalPrefixSumConstantsLayer3        = [ self getPrivateMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer3"     ];
    }
}


- (void) setInitialMetalConstants
{

    struct radix_sort_constants c;

    memset( &c, (uint)0, sizeof(struct radix_sort_constants) );

    c.total_num_elements = _mNumElements;

    memcpy( _mMetalConstRadixSortConstants.contents, &c, sizeof(struct radix_sort_constants) );


    if ( _mPrefixSumConfiguration == 1 || _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer1.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer1.contents, &c, sizeof(struct prefix_sum_constants) );
    }

    if ( _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer2.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer2.contents, &c, sizeof(struct prefix_sum_constants) );
    }

    if ( _mPrefixSumConfiguration == 3 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer3.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer3.contents, &c, sizeof(struct prefix_sum_constants) );
    }
}


- (void) createMetalPipelineStates
{
    [ self loadLibraryWithName:@"./radix_sort.metallib" ];

    _mPSO_four_way_prefix_sum_with_inblock_shuffle        = [ self getPipelineStateForFunction: @"four_way_prefix_sum_with_inblock_shuffle" ];
    if ( _mCoalescedWrite  ) {
        _mPSO_coalesced_block_mapping_for_the_n_chunk_input = [ self getPipelineStateForFunction: @"coalesced_block_mapping_for_the_n_chunk_input" ];
    }
    else {
        _mPSO_uncoalesced_block_mapping_for_the_n_chunk_input = [ self getPipelineStateForFunction: @"uncoalesced_block_mapping_for_the_n_chunk_input" ];
    }
    _mPSO_is_sorted_within_threadgroups                   = [ self getPipelineStateForFunction: @"is_sorted_within_threadgroups" ];
    _mPSO_are_all_less_than_equal                         = [ self getPipelineStateForFunction: @"are_all_less_than_equal" ];
    _mPSO_scan_threadgroupwise_intermediate_32_32         = [ self getPipelineStateForFunction: @"scan_threadgroupwise_intermediate_32_32" ];
    _mPSO_sum_threadgroup_32_32                           = [ self getPipelineStateForFunction: @"sum_threadgroup_32_32" ];
    _mPSO_scan_with_base_threadgroupwise_32_32            = [ self getPipelineStateForFunction: @"scan_with_base_threadgroupwise_32_32" ];
}


- (instancetype) initWithNumElements:(size_t) num_elements forFloat:(bool) for_float CoalescedWrite:(bool) coalesced_write EarlyOut:(bool) early_out
{
    self = [super init];
    if (self) {

        _mResultOn1       = false;
        _mNumElements     = num_elements;
        _mNumThreadgroups = alignUpAndDivide( num_elements, 1024 );
        _mForFloat        = for_float;
        _mCoalescedWrite  = coalesced_write;
        _mEarlyOut        = early_out;

        [ self createMetalPipelineStates ];

        [ self prefixSumFindConfiguration ];

        [ self allocateMetalBuffers ];

        [ self setInitialMetalConstants ];
    }    
    return self;
}

- (void) resetBufferFlag
{
    _mResultOn1 = false;
}

- (uint) numElements
{
    return _mNumElements;
}

- (int*) getRawPointerIn
{
    if ( _mResultOn1 ) {
        return (int*)_mMetalArray2.contents;
    }
    else {
        return (int*)_mMetalArray1.contents;
    }
}

- (int*) getRawPointerOut
{
    if ( _mResultOn1 ) {
        return (int*)_mMetalArray1.contents;
    }
    else {
        return (int*)_mMetalArray2.contents;
    }
}

- (int*) getRawPointerIn1
{
    return (int*)_mMetalArray1.contents;
}

- (int*) getRawPointerIn2
{
    return (int*)_mMetalArray2.contents;
}


- (void) encodeMetalPrefixSumForEncoder:(id<MTLComputeCommandEncoder>) encoder Buffer:(id<MTLBuffer>) inout_buffer
{
    if ( _mPrefixSumConfiguration == 1 ) {

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:0 ];
        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];
    }

    else if ( _mPrefixSumConfiguration == 2 ) {

        [ encoder setComputePipelineState: _mPSO_sum_threadgroup_32_32 ];

        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1     offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2 offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_32 ];

        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:0 ];
        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];
    }
    else if ( _mPrefixSumConfiguration == 3 ) {

        [ encoder setComputePipelineState: _mPSO_sum_threadgroup_32_32 ];

        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1     offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2 offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2     offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_32 ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2 offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2 offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer3 offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer3     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer3.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer3.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_32 ];


        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2 offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer3     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:0 ];
        [ encoder setBuffer: inout_buffer                        offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1 offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_ atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];
    }
}


- (bool) isArraySorted
{
    struct parallel_order_checking_constants c;
    memset( &c, (uint)0, sizeof(struct parallel_order_checking_constants) );
    c.total_num_elements = _mNumElements;
    memcpy( _mMetalConstParallelOrderCheckingConstants.contents, &c, sizeof(struct parallel_order_checking_constants) );

    // Reset the atomic flag.
    ((int*)_mMetalArrayIsUnsorted.contents)[0] = 0;

    id<MTLCommandBuffer> metal_command_buffer = [ self.commandQueue commandBuffer ];

    assert( metal_command_buffer != nil );

    id<MTLComputeCommandEncoder> metal_encoder = [ metal_command_buffer computeCommandEncoder ];

    assert( metal_encoder != nil );

    [ metal_encoder setComputePipelineState: _mPSO_is_sorted_within_threadgroups ];

    if ( _mResultOn1 ) {
        [ metal_encoder setBuffer: _mMetalArray2 offset:0 atIndex:0 ];
    }
    else {
        [ metal_encoder setBuffer: _mMetalArray1 offset:0 atIndex:0 ];
    }
    [ metal_encoder setBuffer: _mMetalArrayIsUnsorted                     offset:0 atIndex:1 ];
    [ metal_encoder setBuffer: _mMetalThreadgroupBoundariesPrevLast       offset:0 atIndex:2 ];
    [ metal_encoder setBuffer: _mMetalThreadgroupBoundariesFirst          offset:0 atIndex:3 ];
    [ metal_encoder setBuffer: _mMetalConstParallelOrderCheckingConstants offset:0 atIndex:4 ];

    [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(              1024, 1, 1) ];

    [ metal_encoder endEncoding ];

    [ metal_command_buffer commit ];

    [ metal_command_buffer waitUntilCompleted ];

    if ( ((int*)_mMetalArrayIsUnsorted.contents)[0] != 0 ) {
        return false;
    }

    if ( _mNumThreadgroups <= 1 ) {
        return true;
    }
    return false;

    const uint num_elements2         = alignUpAndDivide( _mNumElements, 1024 );
    const uint num_mNumThreadgroups2 = alignUpAndDivide( num_elements2, 1024 );
    const uint num_threads2          = ( num_elements2 < 1024 ) ? num_elements2 : 1024;

    memset( &c, (uint)0, sizeof(struct parallel_order_checking_constants) );
    c.total_num_elements = num_elements2;
    memcpy( _mMetalConstParallelOrderCheckingConstants.contents, &c, sizeof(struct parallel_order_checking_constants) );

    id<MTLCommandBuffer> metal_command_buffer2 = [ self.commandQueue commandBuffer ];

    assert( metal_command_buffer2 != nil );

    id<MTLComputeCommandEncoder> metal_encoder2 = [ metal_command_buffer2 computeCommandEncoder ];

    assert( metal_encoder2 != nil );

    [ metal_encoder2 setComputePipelineState: _mPSO_are_all_less_than_equal ];

    [ metal_encoder2 setBuffer: _mMetalThreadgroupBoundariesPrevLast        offset:0 atIndex:0 ];
    [ metal_encoder2 setBuffer: _mMetalArrayIsUnsorted                      offset:0 atIndex:1 ];
    [ metal_encoder2 setBuffer: _mMetalThreadgroupBoundariesFirst           offset:0 atIndex:2 ];
    [ metal_encoder2 setBuffer: _mMetalConstParallelOrderCheckingConstants  offset:0 atIndex:3 ];

    c.total_num_elements = _mNumThreadgroups;
    [ metal_encoder2 dispatchThreadgroups:MTLSizeMake( num_mNumThreadgroups2, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(          num_threads2, 1, 1) ];

    [ metal_encoder2 endEncoding ];

    [ metal_command_buffer2 commit ];

    [ metal_command_buffer2 waitUntilCompleted ];

    if ( ((int*)_mMetalArrayIsUnsorted.contents)[0] != 0 ) {
        return false;
    }
    else {
        return true;
    }
}

- (void) performComputationForOneShift:(uint)shift
{

    struct radix_sort_constants c;
    memset( &c, (uint)0, sizeof(struct radix_sort_constants) );
    c.total_num_elements = _mNumElements;
    c.bit_right_shift = shift;
    c.flip_msb = (shift == 30)?true:false;
    c.for_float = _mForFloat;

    memcpy( _mMetalConstRadixSortConstants.contents, &c, sizeof(struct radix_sort_constants) );

    id<MTLCommandBuffer> metal_command_buffer = [ self.commandQueue commandBuffer ];

    assert( metal_command_buffer != nil );

    id<MTLComputeCommandEncoder> metal_encoder = [ metal_command_buffer computeCommandEncoder ];

    assert( metal_encoder != nil );

    [ metal_encoder setComputePipelineState: _mPSO_four_way_prefix_sum_with_inblock_shuffle ];

    if ( _mResultOn1 ) {
        [ metal_encoder setBuffer: _mMetalArray2 offset:0 atIndex:0 ];
    }
    else {
        [ metal_encoder setBuffer: _mMetalArray1 offset:0 atIndex:0 ];
    }
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane0 offset:0 atIndex:1 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane1 offset:0 atIndex:2 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane2 offset:0 atIndex:3 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane3 offset:0 atIndex:4 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane1 offset:0 atIndex:5 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane2 offset:0 atIndex:6 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane3 offset:0 atIndex:7 ];

    [ metal_encoder setBuffer: _mMetalConstRadixSortConstants        offset:0 atIndex:8 ];
    [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(              1024, 1, 1) ];

    [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ self encodeMetalPrefixSumForEncoder:metal_encoder Buffer:_mMetalPartialSumsPerThreadgroupLane0 ];
    [ self encodeMetalPrefixSumForEncoder:metal_encoder Buffer:_mMetalPartialSumsPerThreadgroupLane1 ];
    [ self encodeMetalPrefixSumForEncoder:metal_encoder Buffer:_mMetalPartialSumsPerThreadgroupLane2 ];
    [ self encodeMetalPrefixSumForEncoder:metal_encoder Buffer:_mMetalPartialSumsPerThreadgroupLane3 ];

    [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    if ( _mCoalescedWrite  ) {
        [ metal_encoder setComputePipelineState: _mPSO_coalesced_block_mapping_for_the_n_chunk_input ];
    }
    else {
        [ metal_encoder setComputePipelineState: _mPSO_uncoalesced_block_mapping_for_the_n_chunk_input ];
    }
    if ( _mResultOn1 ) {
        [ metal_encoder setBuffer: _mMetalArray2 offset:0 atIndex:0 ];
        [ metal_encoder setBuffer: _mMetalArray1 offset:0 atIndex:1 ];
    }
    else {
        [ metal_encoder setBuffer: _mMetalArray1 offset:0 atIndex:0 ];
        [ metal_encoder setBuffer: _mMetalArray2 offset:0 atIndex:1 ];
    }
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane0 offset:0 atIndex:2 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane1 offset:0 atIndex:3 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane2 offset:0 atIndex:4 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane3 offset:0 atIndex:5 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane1 offset:0 atIndex:6 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane2 offset:0 atIndex:7 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane3 offset:0 atIndex:8 ];

    [ metal_encoder setBuffer: _mMetalConstRadixSortConstants        offset:0 atIndex:9 ];

    [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(              1024, 1, 1) ];

    [ metal_encoder endEncoding ];

    [ metal_command_buffer commit ];

    [ metal_command_buffer waitUntilCompleted ];

}

- (void) performComputation
{
    for ( int i = 0; i < 16; i++ ) {

        _mResultOn1 = ( (i%2) == 0 ) ? false : true;

        if (_mEarlyOut) {

            if ( [ self isArraySorted ] ) {

                _mResultOn1 = ! _mResultOn1;

                // early out
                NSLog(@"Early out at %d.", i );
                return;
            }
        }
        [ self performComputationForOneShift: i*2 ];
    }
}


@end

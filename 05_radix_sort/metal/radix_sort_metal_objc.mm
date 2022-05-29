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
//     else if num elements <= X:
//     then
//         scan_threadgroupwise_intermediate_32_X()
// 
//     else if X < num elements <= X*X:
//     then
//         sum_threadgroup_32_X()
//         scan_threadgroupwise_intermediate_32_X()
//         scan_with_base_threadgroupwise_32_X()
//
//     else if X*X < num elements <= X*X*X:
//     then
//         sum_threadgroup_32_X()
//         sum_threadgroup_32_X()
//         scan_threadgroupwise_intermediate_32_X()
//         scan_with_base_threadgroupwise_32_X()
//         scan_with_base_threadgroupwise_32_X()
//     else if X*X*X < num elements <= X*X*X*X:
//     then
//         sum_threadgroup_32_X()
//         sum_threadgroup_32_X()
//         sum_threadgroup_32_X()
//         scan_threadgroupwise_intermediate_32_X()
//         scan_with_base_threadgroupwise_32_X()
//         scan_with_base_threadgroupwise_32_X()
//         scan_with_base_threadgroupwise_32_X()
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
    int                         _mNumIterationsPerCommit;

    size_t                      _mNumThreadsPerThreadgroup;
    uint                        _mPrefixSumConfiguration;

    PrefixSumLayerParams        _mPrefixSumParamsLayer1;
    PrefixSumLayerParams        _mPrefixSumParamsLayer2;
    PrefixSumLayerParams        _mPrefixSumParamsLayer3;
    PrefixSumLayerParams        _mPrefixSumParamsLayer4;

    id<MTLComputePipelineState> _mPSO_four_way_prefix_sum_with_inblock_shuffle;
    id<MTLComputePipelineState> _mPSO_coalesced_block_mapping_for_the_n_chunk_input;
    id<MTLComputePipelineState> _mPSO_uncoalesced_block_mapping_for_the_n_chunk_input;

    id<MTLComputePipelineState> _mPSO_is_sorted_within_threadgroups;
    id<MTLComputePipelineState> _mPSO_are_all_less_than_equal;

    id<MTLComputePipelineState> _mPSO_scan_threadgroupwise_intermediate_32_X;
    id<MTLComputePipelineState> _mPSO_sum_threadgroup_32_X;
    id<MTLComputePipelineState> _mPSO_scan_with_base_threadgroupwise_32_X;

    id<MTLBuffer> _mMetalArray1;
    id<MTLBuffer> _mMetalArray2;
    id<MTLBuffer> _mMetalThreadgroupBoundariesPrevLast;
    id<MTLBuffer> _mMetalThreadgroupBoundariesFirst;
    id<MTLBuffer> _mMetalArrayIsUnsorted;

    id<MTLBuffer> _mMetalConstRadixSortConstants;
    id<MTLBuffer> _mMetalConstParallelOrderCheckingConstants;

    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane0In;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane0Out;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane1In;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane1Out;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane2In;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane2Out;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane3In;
    id<MTLBuffer> _mMetalPartialSumsPerThreadgroupLane3Out;

    id<MTLBuffer> _mMetalStartPosWithinThreadgroupLane1;
    id<MTLBuffer> _mMetalStartPosWithinThreadgroupLane2;
    id<MTLBuffer> _mMetalStartPosWithinThreadgroupLane3;

    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer1In;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer1Out;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer2In;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer2Out;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer3In;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer3Out;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer4In;
    id<MTLBuffer> _mMetalPrefixSumGridPrefixSumLayer4Out;

    id<MTLBuffer> _mMetalPrefixSumConstantsLayer1;
    id<MTLBuffer> _mMetalPrefixSumConstantsLayer2;
    id<MTLBuffer> _mMetalPrefixSumConstantsLayer3;
    id<MTLBuffer> _mMetalPrefixSumConstantsLayer4;

    id<MTLBuffer> _mMetalConstRadixSortConstants_01;
    id<MTLBuffer> _mMetalConstRadixSortConstants_02;
    id<MTLBuffer> _mMetalConstRadixSortConstants_03;
    id<MTLBuffer> _mMetalConstRadixSortConstants_04;
    id<MTLBuffer> _mMetalConstRadixSortConstants_05;
    id<MTLBuffer> _mMetalConstRadixSortConstants_06;
    id<MTLBuffer> _mMetalConstRadixSortConstants_07;
    id<MTLBuffer> _mMetalConstRadixSortConstants_08;
    id<MTLBuffer> _mMetalConstRadixSortConstants_09;
    id<MTLBuffer> _mMetalConstRadixSortConstants_10;
    id<MTLBuffer> _mMetalConstRadixSortConstants_11;
    id<MTLBuffer> _mMetalConstRadixSortConstants_12;
    id<MTLBuffer> _mMetalConstRadixSortConstants_13;
    id<MTLBuffer> _mMetalConstRadixSortConstants_14;
    id<MTLBuffer> _mMetalConstRadixSortConstants_15;
    id<MTLBuffer> _mMetalConstRadixSortConstants_16;
}


-(void) prefixSumFindConfiguration
{ 
    const uint num_elems_layer1 = alignUpAndDivide( _mNumElements,    _mNumThreadsPerThreadgroup );
    const uint num_elems_layer2 = alignUpAndDivide( num_elems_layer1, _mNumThreadsPerThreadgroup );
    const uint num_elems_layer3 = alignUpAndDivide( num_elems_layer2, _mNumThreadsPerThreadgroup );
    const uint num_elems_layer4 = alignUpAndDivide( num_elems_layer3, _mNumThreadsPerThreadgroup );

    if ( num_elems_layer1 <= _mNumThreadsPerThreadgroup ) {

        _mPrefixSumConfiguration = 1;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, alignUp( num_elems_layer1, 32 ), 1 );
    }
    else if  ( num_elems_layer1 <= _mNumThreadsPerThreadgroup * _mNumThreadsPerThreadgroup ) {

        _mPrefixSumConfiguration = 2;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, _mNumThreadsPerThreadgroup,      num_elems_layer2 );
        _mPrefixSumParamsLayer2.set( num_elems_layer2, alignUp( num_elems_layer2, 32 ), 1                );
    }
    else if  ( num_elems_layer1 <= _mNumThreadsPerThreadgroup * _mNumThreadsPerThreadgroup * _mNumThreadsPerThreadgroup ) {

        _mPrefixSumConfiguration = 3;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, _mNumThreadsPerThreadgroup,      num_elems_layer2 );
        _mPrefixSumParamsLayer2.set( num_elems_layer2, _mNumThreadsPerThreadgroup,      num_elems_layer3 );
        _mPrefixSumParamsLayer3.set( num_elems_layer3, alignUp( num_elems_layer3, 32 ), 1                );
    }
    else if  ( num_elems_layer1 <= _mNumThreadsPerThreadgroup * _mNumThreadsPerThreadgroup * _mNumThreadsPerThreadgroup * _mNumThreadsPerThreadgroup ) {

        _mPrefixSumConfiguration = 4;

        _mPrefixSumParamsLayer1.set( num_elems_layer1, _mNumThreadsPerThreadgroup,      num_elems_layer2 );
        _mPrefixSumParamsLayer2.set( num_elems_layer2, _mNumThreadsPerThreadgroup,      num_elems_layer3 );
        _mPrefixSumParamsLayer3.set( num_elems_layer3, _mNumThreadsPerThreadgroup,      num_elems_layer4 );
        _mPrefixSumParamsLayer4.set( num_elems_layer4, alignUp( num_elems_layer4, 32 ), 1                );
    }
    else {
        _mPrefixSumConfiguration = 0;
    }

//    NSLog( @"_mPrefixSumParamsLayer1.num_elements_:                %d", _mPrefixSumParamsLayer1.num_elements_                );
//    NSLog( @"_mPrefixSumParamsLayer1.num_threads_per_threadgroup_: %d", _mPrefixSumParamsLayer1.num_threads_per_threadgroup_ );
//    NSLog( @"_mPrefixSumParamsLayer1.num_threadgroups_per_grid_:   %d", _mPrefixSumParamsLayer1.num_threadgroups_per_grid_   );
//    NSLog( @"_mPrefixSumParamsLayer2.num_elements_:                %d", _mPrefixSumParamsLayer2.num_elements_                );
//    NSLog( @"_mPrefixSumParamsLayer2.num_threads_per_threadgroup_: %d", _mPrefixSumParamsLayer2.num_threads_per_threadgroup_ );
//    NSLog( @"_mPrefixSumParamsLayer2.num_threadgroups_per_grid_:   %d", _mPrefixSumParamsLayer2.num_threadgroups_per_grid_   );
//    NSLog( @"_mPrefixSumParamsLayer3.num_elements_:                %d", _mPrefixSumParamsLayer3.num_elements_                );
//    NSLog( @"_mPrefixSumParamsLayer3.num_threads_per_threadgroup_: %d", _mPrefixSumParamsLayer3.num_threads_per_threadgroup_ );
//    NSLog( @"_mPrefixSumParamsLayer3.num_threadgroups_per_grid_:   %d", _mPrefixSumParamsLayer3.num_threadgroups_per_grid_   );
//    NSLog( @"_mPrefixSumParamsLayer4.num_elements_:                %d", _mPrefixSumParamsLayer4.num_elements_                );
//    NSLog( @"_mPrefixSumParamsLayer4.num_threads_per_threadgroup_: %d", _mPrefixSumParamsLayer4.num_threads_per_threadgroup_ );
//    NSLog( @"_mPrefixSumParamsLayer4.num_threadgroups_per_grid_:   %d", _mPrefixSumParamsLayer4.num_threadgroups_per_grid_   );
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
    _mMetalPartialSumsPerThreadgroupLane0In    = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)          for:@"_mMetalPartialSumsPerThreadgroupLane0In"     ];
    _mMetalPartialSumsPerThreadgroupLane0Out   = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)          for:@"_mMetalPartialSumsPerThreadgroupLane0Out"    ];
    _mMetalPartialSumsPerThreadgroupLane1In    = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)           for:@"_mMetalPartialSumsPerThreadgroupLane1In"    ];
    _mMetalPartialSumsPerThreadgroupLane1Out   = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)           for:@"_mMetalPartialSumsPerThreadgroupLane1Out"   ];
    _mMetalPartialSumsPerThreadgroupLane2In    = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)           for:@"_mMetalPartialSumsPerThreadgroupLane2In"    ];
    _mMetalPartialSumsPerThreadgroupLane2Out   = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)           for:@"_mMetalPartialSumsPerThreadgroupLane2Out"   ];
    _mMetalPartialSumsPerThreadgroupLane3In    = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)           for:@"_mMetalPartialSumsPerThreadgroupLane3In"    ];
    _mMetalPartialSumsPerThreadgroupLane3Out   = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(int)* _mNumThreadgroups, 16)           for:@"_mMetalPartialSumsPerThreadgroupLane3Out"   ];
    _mMetalStartPosWithinThreadgroupLane1      = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(unsigned short)*_mNumThreadgroups, 16) for:@"_mMetalStartPosWithinThreadgroupLane1"      ];
    _mMetalStartPosWithinThreadgroupLane2      = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(unsigned short)*_mNumThreadgroups, 16) for:@"_mMetalStartPosWithinThreadgroupLane2"      ];
    _mMetalStartPosWithinThreadgroupLane3      = [ self getPrivateMTLBufferForBytes: alignUp(sizeof(unsigned short)*_mNumThreadgroups, 16) for:@"_mMetalStartPosWithinThreadgroupLane3"      ];

    if ( _mPrefixSumConfiguration == 1 || _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 || _mPrefixSumConfiguration == 4 ) {

        _mMetalPrefixSumGridPrefixSumLayer1In  = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer1.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer1In" ];
        _mMetalPrefixSumGridPrefixSumLayer1Out = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer1.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer1Out" ];
        _mMetalPrefixSumConstantsLayer1        = [ self getSharedMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer1"     ];
    }
    if ( _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 || _mPrefixSumConfiguration == 4 ) {

        _mMetalPrefixSumGridPrefixSumLayer2In  = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer2.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer2In" ];
        _mMetalPrefixSumGridPrefixSumLayer2Out = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer2.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer2Out" ];
        _mMetalPrefixSumConstantsLayer2        = [ self getSharedMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer2"     ];
    }
    if ( _mPrefixSumConfiguration == 3 || _mPrefixSumConfiguration == 4 ) {

        _mMetalPrefixSumGridPrefixSumLayer3In  = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer3.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer3In" ];
        _mMetalPrefixSumGridPrefixSumLayer3Out = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer3.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer3Out" ];
        _mMetalPrefixSumConstantsLayer3        = [ self getSharedMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer3"     ];
    }
    if ( _mPrefixSumConfiguration == 4 ) {

        _mMetalPrefixSumGridPrefixSumLayer4In  = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer4.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer4In" ];
        _mMetalPrefixSumGridPrefixSumLayer4Out = [ self getPrivateMTLBufferForBytes: sizeof(int)*_mPrefixSumParamsLayer4.num_threadgroups_per_grid_
                                                                                for: @"_mMetalPrefixSumGridPrefixSumLayer4Out" ];
        _mMetalPrefixSumConstantsLayer4        = [ self getSharedMTLBufferForBytes: sizeof(struct prefix_sum_constants)
                                                                                for: @"_mMetalPrefixSumConstantsLayer4"     ];
    }

    _mMetalConstRadixSortConstants_01          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_01" ];
    _mMetalConstRadixSortConstants_02          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_02" ];
    _mMetalConstRadixSortConstants_03          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_03" ];
    _mMetalConstRadixSortConstants_04          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_04" ];
    _mMetalConstRadixSortConstants_05          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_05" ];
    _mMetalConstRadixSortConstants_06          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_06" ];
    _mMetalConstRadixSortConstants_07          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_07" ];
    _mMetalConstRadixSortConstants_08          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_08" ];
    _mMetalConstRadixSortConstants_09          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_09" ];
    _mMetalConstRadixSortConstants_10          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_10" ];
    _mMetalConstRadixSortConstants_11          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_11" ];
    _mMetalConstRadixSortConstants_12          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_12" ];
    _mMetalConstRadixSortConstants_13          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_13" ];
    _mMetalConstRadixSortConstants_14          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_14" ];
    _mMetalConstRadixSortConstants_15          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_15" ];
    _mMetalConstRadixSortConstants_16          = [ self getSharedMTLBufferForBytes:  sizeof(struct radix_sort_constants)      for:@"_mMetalConstRadixSortConstants_16" ];
}


- (void) setInitialMetalConstants
{

    struct radix_sort_constants c;

    memset( &c, (uint)0, sizeof(struct radix_sort_constants) );

    c.total_num_elements = _mNumElements;

    memcpy( _mMetalConstRadixSortConstants.contents, &c, sizeof(struct radix_sort_constants) );


    if ( _mPrefixSumConfiguration == 1 || _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 || _mPrefixSumConfiguration == 4 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer1.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer1.contents, &c, sizeof(struct prefix_sum_constants) );
    }

    if ( _mPrefixSumConfiguration == 2 || _mPrefixSumConfiguration == 3 || _mPrefixSumConfiguration == 4 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer2.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer2.contents, &c, sizeof(struct prefix_sum_constants) );
    }

    if ( _mPrefixSumConfiguration == 3 || _mPrefixSumConfiguration == 4 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer3.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer3.contents, &c, sizeof(struct prefix_sum_constants) );
    }

    if ( _mPrefixSumConfiguration == 4 ) {

            struct prefix_sum_constants c;

            memset( &c, (uint)0, sizeof(struct prefix_sum_constants) );

            c.num_elements = _mPrefixSumParamsLayer4.num_elements_;

            memcpy( _mMetalPrefixSumConstantsLayer4.contents, &c, sizeof(struct prefix_sum_constants) );
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
    if ( _mForFloat ) {
        _mPSO_is_sorted_within_threadgroups                   = [ self getPipelineStateForFunction: @"is_sorted_within_threadgroups_float" ];
        _mPSO_are_all_less_than_equal                         = [ self getPipelineStateForFunction: @"are_all_less_than_equal_float" ];
    }
    else {
        _mPSO_is_sorted_within_threadgroups                   = [ self getPipelineStateForFunction: @"is_sorted_within_threadgroups_int" ];
        _mPSO_are_all_less_than_equal                         = [ self getPipelineStateForFunction: @"are_all_less_than_equal_int" ];
    }
    _mPSO_scan_threadgroupwise_intermediate_32_X          = [ self getPipelineStateForFunction: @"scan_threadgroupwise_intermediate_32_X_int" ];
    _mPSO_sum_threadgroup_32_X                            = [ self getPipelineStateForFunction: @"sum_threadgroup_32_X_int" ];
    _mPSO_scan_with_base_threadgroupwise_32_X             = [ self getPipelineStateForFunction: @"scan_with_base_threadgroupwise_32_X_int" ];
}


- (instancetype) initWithNumElements:(size_t)  num_elements 
                            forFloat:(bool)    for_float 
                      CoalescedWrite:(bool)    coalesced_write 
                            EarlyOut:(bool)    early_out
              NumIterationsPerCommit:(int)     num_iterations_per_commit
           NumThreadsPerThreadgrouop:(size_t)  num_threads_per_threadgroup
{
    self = [super init];
    if (self) {

        _mResultOn1                = false;
        _mNumElements              = num_elements;
        _mNumThreadgroups          = alignUpAndDivide( num_elements, num_threads_per_threadgroup );
        _mForFloat                 = for_float;
        _mCoalescedWrite           = coalesced_write;
        _mEarlyOut                 = early_out;
        _mNumIterationsPerCommit   = num_iterations_per_commit;
        _mNumThreadsPerThreadgroup = num_threads_per_threadgroup;

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


- (void) encodeMetalPrefixSumForEncoder:(id<MTLComputeCommandEncoder>) encoder InBuffer:(id<MTLBuffer>) in_buffer  OutBuffer:(id<MTLBuffer>) out_buffer
{
    if ( _mPrefixSumConfiguration == 1 ) {

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_X ];

        [ encoder setBuffer: in_buffer                             offset:0 atIndex:0 ];
        [ encoder setBuffer: out_buffer                            offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1       offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];
    }

    else if ( _mPrefixSumConfiguration == 2 ) {

        [ encoder setComputePipelineState: _mPSO_sum_threadgroup_32_X ];

        [ encoder setBuffer: in_buffer                             offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1       offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_X ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1Out offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2In  offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_X ];

        [ encoder setBuffer: in_buffer                              offset:0 atIndex:0 ];
        [ encoder setBuffer: out_buffer                             offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1Out offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1        offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];
    }
    else if ( _mPrefixSumConfiguration == 3 ) {

        [ encoder setComputePipelineState: _mPSO_sum_threadgroup_32_X ];

        [ encoder setBuffer: in_buffer                             offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1       offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2In  offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2        offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_X ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2Out offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer3In  offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer3     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer3.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_X ];


        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1Out offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2Out offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: in_buffer                              offset:0 atIndex:0 ];
        [ encoder setBuffer: out_buffer                             offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1Out offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1        offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];
    }
    else if ( _mPrefixSumConfiguration == 4 ) {

        [ encoder setComputePipelineState: _mPSO_sum_threadgroup_32_X ];

        [ encoder setBuffer: in_buffer                             offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1       offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer1.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2In  offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2        offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer3In  offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer3        offset:0 atIndex:2 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer3.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_threadgroupwise_intermediate_32_X ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer3In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer3Out offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer4In  offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer4     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer4.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer4.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer4.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setComputePipelineState: _mPSO_scan_with_base_threadgroupwise_32_X ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2Out offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer3Out offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer3     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer3.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer3.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1In  offset:0 atIndex:0 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1Out offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer2Out offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer2     offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 16) atIndex:0 ];

        [ encoder dispatchThreadgroups:MTLSizeMake( _mPrefixSumParamsLayer2.num_threadgroups_per_grid_,   1, 1 )
                 threadsPerThreadgroup:MTLSizeMake( _mPrefixSumParamsLayer2.num_threads_per_threadgroup_, 1, 1 ) ];

        [ encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ encoder setBuffer: in_buffer                              offset:0 atIndex:0 ];
        [ encoder setBuffer: out_buffer                             offset:0 atIndex:1 ];
        [ encoder setBuffer: _mMetalPrefixSumGridPrefixSumLayer1Out offset:0 atIndex:2 ];
        [ encoder setBuffer: _mMetalPrefixSumConstantsLayer1        offset:0 atIndex:3 ];

        [ encoder setThreadgroupMemoryLength: alignUp(sizeof(int) * _mPrefixSumParamsLayer1.num_threads_per_threadgroup_, 16) atIndex:0 ];

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

    [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups,          1, 1)
                   threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];
 
    [ metal_encoder endEncoding ];

    [ metal_command_buffer commit ];

    [ metal_command_buffer waitUntilCompleted ];

    if ( ((int*)_mMetalArrayIsUnsorted.contents)[0] != 0 ) {
        return false;
    }

    if ( _mNumThreadgroups <= 1 ) {
        return true;
    }

    const uint num_elements2         = alignUpAndDivide( _mNumElements, _mNumThreadsPerThreadgroup );
    const uint num_mNumThreadgroups2 = alignUpAndDivide( num_elements2, _mNumThreadsPerThreadgroup );
    const uint num_threads2          = ( num_elements2 < _mNumThreadsPerThreadgroup ) ? num_elements2 : _mNumThreadsPerThreadgroup;

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
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane0In offset:0 atIndex:  1 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane1In offset:0 atIndex:  2 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane2In offset:0 atIndex:  3 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane3In offset:0 atIndex:  4 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane1   offset:0 atIndex:  5 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane2   offset:0 atIndex:  6 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane3   offset:0 atIndex:  7 ];
    [ metal_encoder setBuffer: _mMetalConstRadixSortConstants          offset:0 atIndex:  8 ];

    [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:0 ]; // lane_counts_lane0
    [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:1 ]; // lane_counts_lane1
    [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:2 ]; // lane_counts_lane2
    [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:3 ]; // lane_counts_lane3
    [ metal_encoder setThreadgroupMemoryLength: sizeof(int)            * _mNumThreadsPerThreadgroup atIndex:4 ]; // target_array

    [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups,          1, 1)
                   threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];

    [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane0In OutBuffer:_mMetalPartialSumsPerThreadgroupLane0Out ];
    [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane1In OutBuffer:_mMetalPartialSumsPerThreadgroupLane1Out ];
    [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane2In OutBuffer:_mMetalPartialSumsPerThreadgroupLane2Out ];
    [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane3In OutBuffer:_mMetalPartialSumsPerThreadgroupLane3Out ];

    [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

    if ( _mCoalescedWrite  ) {
        [ metal_encoder setComputePipelineState: _mPSO_coalesced_block_mapping_for_the_n_chunk_input ];
        [ metal_encoder setThreadgroupMemoryLength: sizeof(int) * _mNumThreadsPerThreadgroup atIndex:0 ]; // copy_src_array
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
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane0Out offset:0 atIndex:2 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane1Out offset:0 atIndex:3 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane2Out offset:0 atIndex:4 ];
    [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane3Out offset:0 atIndex:5 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane1    offset:0 atIndex:6 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane2    offset:0 atIndex:7 ];
    [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane3    offset:0 atIndex:8 ];

    [ metal_encoder setBuffer: _mMetalConstRadixSortConstants        offset:0 atIndex:9 ];

    [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups,          1, 1)
                   threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];

    [ metal_encoder endEncoding ];

    [ metal_command_buffer commit ];

    [ metal_command_buffer waitUntilCompleted ];

}

- (void) performComputation
{
        
    if ( _mNumIterationsPerCommit > 1 ) {
        [ self performComputationInFewerCommits ];
    }
    else {

        for ( int i = 0; i < 16; i++ ) {

            _mResultOn1 = ( (i%2) == 0 ) ? false : true;

            if (_mEarlyOut) {

                if ( [ self isArraySorted ] ) {

                    // early out
                    NSLog(@"Early out at %d.", i );

                    // NOTE: comment out 'return' for time measurements.
                    // _mResultOn1 = ! _mResultOn1;
                    //return;
                }
            }
            [ self performComputationForOneShift: i*2 ];
        }
    }
}

- (void) performComputationInFewerCommits
{
    for ( int i = 0; i < 16; i++ ) {

        const int shift = i*2;
        struct radix_sort_constants c;

        memset( &c, (uint)0, sizeof(struct radix_sort_constants) );

        c.total_num_elements = _mNumElements;
        c.bit_right_shift    = shift;
        c.flip_msb           = (shift == 30)?true:false;
        c.for_float          = _mForFloat;

        switch (i) {

          case 0:
            memcpy( _mMetalConstRadixSortConstants_01.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 1:
            memcpy( _mMetalConstRadixSortConstants_02.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 2:
            memcpy( _mMetalConstRadixSortConstants_03.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 3:
            memcpy( _mMetalConstRadixSortConstants_04.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 4:
            memcpy( _mMetalConstRadixSortConstants_05.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 5:
            memcpy( _mMetalConstRadixSortConstants_06.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 6:
            memcpy( _mMetalConstRadixSortConstants_07.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 7:
            memcpy( _mMetalConstRadixSortConstants_08.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 8:
            memcpy( _mMetalConstRadixSortConstants_09.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 9:
            memcpy( _mMetalConstRadixSortConstants_10.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 10:
            memcpy( _mMetalConstRadixSortConstants_11.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 11:
            memcpy( _mMetalConstRadixSortConstants_12.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 12:
            memcpy( _mMetalConstRadixSortConstants_13.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 13:
            memcpy( _mMetalConstRadixSortConstants_14.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 14:
            memcpy( _mMetalConstRadixSortConstants_15.contents, &c, sizeof(struct radix_sort_constants) );
            break;

          case 15:
          default:
            memcpy( _mMetalConstRadixSortConstants_16.contents, &c, sizeof(struct radix_sort_constants) );
            break;
        }
    }


    id<MTLCommandBuffer> metal_command_buffer;

    id<MTLComputeCommandEncoder> metal_encoder;

    for ( int i = 0; i < 16; i++ ) {

        if ( (i % _mNumIterationsPerCommit) == 0 ) {

            metal_command_buffer = [ self.commandQueue commandBuffer ];

            assert( metal_command_buffer != nil );

            metal_encoder = [ metal_command_buffer computeCommandEncoder ];

            assert( metal_encoder != nil );
        }
        
        
        _mResultOn1 = ( (i%2) == 0 ) ? false : true;

        [ metal_encoder setComputePipelineState: _mPSO_four_way_prefix_sum_with_inblock_shuffle ];

        if ( _mResultOn1 ) {
            [ metal_encoder setBuffer: _mMetalArray2 offset:0 atIndex:0 ];
        }
        else {
            [ metal_encoder setBuffer: _mMetalArray1 offset:0 atIndex:0 ];
        }

        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane0In offset:0 atIndex:1 ];
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane1In offset:0 atIndex:2 ];
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane2In offset:0 atIndex:3 ];
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane3In offset:0 atIndex:4 ];
        [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane1   offset:0 atIndex:5 ];
        [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane2   offset:0 atIndex:6 ];
        [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane3   offset:0 atIndex:7 ];

        switch (i) {

          case 0:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_01        offset:0 atIndex:8 ];
            break;

          case 1:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_02        offset:0 atIndex:8 ];
            break;

          case 2:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_03        offset:0 atIndex:8 ];
            break;

          case 3:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_04        offset:0 atIndex:8 ];
            break;

          case 4:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_05        offset:0 atIndex:8 ];
            break;

          case 5:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_06        offset:0 atIndex:8 ];
            break;

          case 6:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_07        offset:0 atIndex:8 ];
            break;

          case 7:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_08        offset:0 atIndex:8 ];
            break;

          case 8:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_09        offset:0 atIndex:8 ];
            break;

          case 9:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_10        offset:0 atIndex:8 ];
            break;

          case 10:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_11        offset:0 atIndex:8 ];
            break;

          case 11:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_12        offset:0 atIndex:8 ];
            break;

          case 12:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_13        offset:0 atIndex:8 ];
            break;

          case 13:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_14        offset:0 atIndex:8 ];
            break;

          case 14:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_15        offset:0 atIndex:8 ];
            break;

          case 15:
          default:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_16        offset:0 atIndex:8 ];
            break;
        }

        [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:0 ]; // lane_counts_lane0
        [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:1 ]; // lane_counts_lane1
        [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:2 ]; // lane_counts_lane2
        [ metal_encoder setThreadgroupMemoryLength: sizeof(unsigned short) * _mNumThreadsPerThreadgroup atIndex:3 ]; // lane_counts_lane3
        [ metal_encoder setThreadgroupMemoryLength: sizeof(int)            * _mNumThreadsPerThreadgroup atIndex:4 ]; // target_array

        [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups,          1, 1)
                       threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];

        [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane0In OutBuffer:_mMetalPartialSumsPerThreadgroupLane0Out ];
        [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane1In OutBuffer:_mMetalPartialSumsPerThreadgroupLane1Out ];
        [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane2In OutBuffer:_mMetalPartialSumsPerThreadgroupLane2Out ];
        [ self encodeMetalPrefixSumForEncoder:metal_encoder InBuffer:_mMetalPartialSumsPerThreadgroupLane3In OutBuffer:_mMetalPartialSumsPerThreadgroupLane3Out ];

        [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];

        if ( _mCoalescedWrite  ) {
            [ metal_encoder setComputePipelineState: _mPSO_coalesced_block_mapping_for_the_n_chunk_input ];
            [ metal_encoder setThreadgroupMemoryLength: sizeof(int) * _mNumThreadsPerThreadgroup atIndex:0 ]; // copy_src_array
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
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane0Out offset:0 atIndex:2 ];
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane1Out offset:0 atIndex:3 ];
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane2Out offset:0 atIndex:4 ];
        [ metal_encoder setBuffer: _mMetalPartialSumsPerThreadgroupLane3Out offset:0 atIndex:5 ];
        [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane1    offset:0 atIndex:6 ];
        [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane2    offset:0 atIndex:7 ];
        [ metal_encoder setBuffer: _mMetalStartPosWithinThreadgroupLane3    offset:0 atIndex:8 ];

        switch (i) {

          case 0:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_01        offset:0 atIndex:9 ];
            break;

          case 1:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_02        offset:0 atIndex:9 ];
            break;

          case 2:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_03        offset:0 atIndex:9 ];
            break;

          case 3:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_04        offset:0 atIndex:9 ];
            break;

          case 4:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_05        offset:0 atIndex:9 ];
            break;

          case 5:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_06        offset:0 atIndex:9 ];
            break;

          case 6:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_07        offset:0 atIndex:9 ];
            break;

          case 7:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_08        offset:0 atIndex:9 ];
            break;

          case 8:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_09        offset:0 atIndex:9 ];
            break;

          case 9:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_10        offset:0 atIndex:9 ];
            break;

          case 10:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_11        offset:0 atIndex:9 ];
            break;

          case 11:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_12        offset:0 atIndex:9 ];
            break;

          case 12:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_13        offset:0 atIndex:9 ];
            break;

          case 13:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_14        offset:0 atIndex:9 ];
            break;

          case 14:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_15        offset:0 atIndex:9 ];
            break;

          case 15:
          default:
            [ metal_encoder setBuffer: _mMetalConstRadixSortConstants_16        offset:0 atIndex:9 ];
            break;
        }

        [ metal_encoder dispatchThreadgroups:MTLSizeMake( _mNumThreadgroups,          1, 1)
                       threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];

        [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ];


        if ( (i % _mNumIterationsPerCommit) == ( _mNumIterationsPerCommit - 1) ) {

            [ metal_encoder endEncoding ];

            [ metal_command_buffer commit ];

            [ metal_command_buffer waitUntilCompleted ];
        }
    }
}

@end

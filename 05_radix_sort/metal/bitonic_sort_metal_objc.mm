#include <memory>
#include <algorithm>

#import "bitonic_sort_metal_objc.h"

static inline uint alignUpAndDivide(uint v, uint d )
{
    return (v + d - 1) / d;
}

typedef struct _bitonic_sort_constants
{
    uint  total_num_elements; // must be 2^x. threadgroups_per_grid must be total_num_elements / 2
    uint  swap_span;
    uint  block_size; // A unit of consecutive sub-array of size 2^x that are sorted in the same direction.
                      // The maximum/initial span for swapping is (2-1)^x.
} bitonic_sort_constants;


@implementation BitonicSortMetalObjC {

    uint                        _mNumElements;
    bool                        _mForFloat;
    size_t                      _mNumThreadsPerThreadgroup;

    id<MTLComputePipelineState> _mPSO_bitonic_sort_one_swap_span;
    id<MTLComputePipelineState> _mPSO_bitonic_sort_multiple_swap_spans_down_to_2;

    id<MTLBuffer> _mMetalBufferConstants;
    id<MTLBuffer> _mMetalBufferTargetArray;
}


- (instancetype) initWithNumElements:(size_t)  num_elements 
                            forFloat:(bool)    for_float 
           NumThreadsPerThreadgrouop:(size_t)  num_threads_per_threadgroup
{
    self = [super init];
    if (self) {

        _mNumElements              = num_elements;
        _mForFloat                 = for_float;
        _mNumThreadsPerThreadgroup = num_threads_per_threadgroup;

        [ self createMetalPipelineStates ];

        [ self allocateMetalBuffers ];
    }    
    return self;
}


- (uint) numElements
{
    return _mNumElements;
}


- (int*) getRawPointerInOut
{
    return (int*)_mMetalBufferTargetArray.contents;
}


- (void) createMetalPipelineStates
{
    [ self loadLibraryWithName:@"./radix_sort.metallib" ];

    if ( _mForFloat ) {
        _mPSO_bitonic_sort_one_swap_span                 = [ self getPipelineStateForFunction: @"bitonic_sort_one_swap_span_float" ];
        _mPSO_bitonic_sort_multiple_swap_spans_down_to_2 = [ self getPipelineStateForFunction: @"bitonic_sort_multiple_swap_spans_down_to_2_float" ];
    }
    else {
        _mPSO_bitonic_sort_one_swap_span                 = [ self getPipelineStateForFunction: @"bitonic_sort_one_swap_span_int" ];
        _mPSO_bitonic_sort_multiple_swap_spans_down_to_2 = [ self getPipelineStateForFunction: @"bitonic_sort_multiple_swap_spans_down_to_2_int" ];
               
    }
}


- (void) allocateMetalBuffers
{
    _mMetalBufferTargetArray = [ self getSharedMTLBufferForBytes: sizeof(int)*_mNumElements      for:@"_mMetalBufferTargetArray" ];
    _mMetalBufferConstants   = [ self getSharedMTLBufferForBytes: sizeof(bitonic_sort_constants) for:@"_mMetalBufferConstants" ];
}


- (void) performComputation
{

    bitonic_sort_constants constants;
    memset( &constants, (uint)0, sizeof(bitonic_sort_constants) );
    constants.total_num_elements = _mNumElements;
    constants.swap_span          =  0;


    id<MTLCommandBuffer> metal_command_buffer = [ self.commandQueue commandBuffer ];

    assert( metal_command_buffer != nil );

    id<MTLComputeCommandEncoder> metal_encoder = [ metal_command_buffer computeCommandEncoder ];

    assert( metal_encoder != nil );

    for ( int block_size = 2; block_size <= _mNumElements; block_size *= 2 ) {

        constants.block_size = block_size;

        int swap_span;

        for ( swap_span = block_size/2; swap_span >= _mNumThreadsPerThreadgroup; swap_span /= 2 ) {

            constants.swap_span =  swap_span;

            [ metal_encoder setComputePipelineState: _mPSO_bitonic_sort_one_swap_span ];
            [ metal_encoder setBuffer: _mMetalBufferTargetArray offset:0 atIndex:0 ];
            [ metal_encoder setBytes: &constants length: sizeof(bitonic_sort_constants) atIndex:1 ];

            const int numThreadgroups =   ( _mNumElements >= _mNumThreadsPerThreadgroup )
                                        ? alignUpAndDivide( _mNumElements, _mNumThreadsPerThreadgroup )
                                        : alignUpAndDivide( 32, _mNumThreadsPerThreadgroup )
                                        ;

            [ metal_encoder dispatchThreadgroups:MTLSizeMake( numThreadgroups,            1, 1)
                           threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];

            [ metal_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers ]; 
        }

        constants.swap_span =  swap_span;

        [ metal_encoder setComputePipelineState: _mPSO_bitonic_sort_multiple_swap_spans_down_to_2 ];
        [ metal_encoder setBuffer: _mMetalBufferTargetArray offset:0 atIndex:0 ];
        [ metal_encoder setBytes: &constants length: sizeof(bitonic_sort_constants) atIndex:1 ];

        const int numThreadgroups =   ( _mNumElements >= _mNumThreadsPerThreadgroup )
                                    ? alignUpAndDivide( _mNumElements, _mNumThreadsPerThreadgroup )
                                    : alignUpAndDivide( 32, _mNumThreadsPerThreadgroup )
                                    ;

        [ metal_encoder dispatchThreadgroups:MTLSizeMake( numThreadgroups,            1, 1)
                       threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerThreadgroup, 1, 1) ];
    }

    [ metal_encoder endEncoding ];

    [ metal_command_buffer commit ];

    [ metal_command_buffer waitUntilCompleted ];
}

@end

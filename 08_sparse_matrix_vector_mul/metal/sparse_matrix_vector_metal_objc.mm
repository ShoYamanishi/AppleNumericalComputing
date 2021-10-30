#include <memory>
#include <algorithm>
#include <iostream>

#import "sparse_matrix_vector_metal_objc.h"


@implementation SparseMatrixVectorMetalObjC
{
    id<MTLComputePipelineState> _mPSO_sparse_matrix_vector_row_per_thread;
    id<MTLComputePipelineState> _mPSO_sparse_matrix_vector_adaptive;

    bool                        _mUSeAdaptation;

    id<MTLBuffer>               _mBlockPtrs;
    id<MTLBuffer>               _mThreadsPerRow;
    id<MTLBuffer>               _mMaxIters;

    id<MTLBuffer>               _mRowPtrs;
    id<MTLBuffer>               _mColumns;
    id<MTLBuffer>               _mValues;
    id<MTLBuffer>               _mVector;
    id<MTLBuffer>               _mOutputVector;
    id<MTLBuffer>               _mConst;

    uint                        _mNumBlocks;
    uint                        _mNumRows;
    uint                        _mNumColumns;
    uint                        _mNumNonzeroElems;
}


- (instancetype) initWithNumRows:(size_t) M
                      NumColumns:(size_t) N
                 NumNonzeroElems:(size_t) num_nonzero_elems
                   UseAdaptation:(bool)   use_adaptation
{
    self = [super init];
    if (self)
    {
        _mNumRows         = M;
        _mNumColumns      = N;
        _mNumNonzeroElems = num_nonzero_elems;
        _mUSeAdaptation   = use_adaptation;

        [ self loadLibraryWithName: @"./sparse_matrix_vector.metallib" ];

        _mPSO_sparse_matrix_vector_row_per_thread
            = [ self getPipelineStateForFunction: @"sparse_matrix_vector_row_per_thread" ];

        _mPSO_sparse_matrix_vector_adaptive
            = [ self getPipelineStateForFunction: @"sparse_matrix_vector_adaptive" ];

        _mBlockPtrs     = [ self getSharedMTLBufferForBytes: sizeof(int)   * (_mNumRows + 1)   for: @"_mBlockPtrs"     ];
        _mThreadsPerRow = [ self getSharedMTLBufferForBytes: sizeof(int)   * (_mNumRows + 1)   for: @"_mThreadsPerRow" ];
        _mMaxIters      = [ self getSharedMTLBufferForBytes: sizeof(int)   * (_mNumRows + 1)   for: @"_mMaxIters"      ];
        _mRowPtrs       = [ self getSharedMTLBufferForBytes: sizeof(int)   * (_mNumRows + 1)   for: @"_mRowPtrs"       ];
        _mColumns       = [ self getSharedMTLBufferForBytes: sizeof(int)   * _mNumNonzeroElems for: @"_mColumns"       ];
        _mValues        = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumNonzeroElems for: @"_mValues"        ];
        _mVector        = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumColumns      for: @"_mVector"        ];
        _mOutputVector  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mNumRows         for: @"_mOutputVector"  ];
        _mConst         = [ self getSharedMTLBufferForBytes: sizeof(struct sparse_matrix_vector_constants) for: @"_mOutputVector" ];
    }
    return self;
}


- (void) setInitialStatesBlockPtrs:(int*)   csr_block_ptrs
                     ThreadsPerRow:(int*)   csr_threads_per_row
                          MaxIters:(int*)   csr_max_iters
                         NumBlocks:(size_t) csr_num_blocks
                           RowPtrs:(int*)   csr_row_ptrs
                           Columns:(int*)   csr_columns
                            Values:(float*) csr_values
                            Vector:(float*) csr_vector
                      OutputVector:(float*) output_vector
{

    _mNumBlocks = csr_num_blocks;

   if ( csr_block_ptrs != nullptr ) {

        memcpy( _mBlockPtrs.contents,     csr_block_ptrs,      sizeof(int) * (_mNumBlocks + 1) );

        memcpy( _mThreadsPerRow.contents, csr_threads_per_row, sizeof(int) * (_mNumBlocks + 1) );

        memcpy( _mMaxIters.contents,      csr_max_iters,       sizeof(int) * (_mNumBlocks + 1) );
   }

    memcpy( _mRowPtrs.contents,   csr_row_ptrs,   sizeof(int)   * (_mNumRows + 1) );

    memcpy( _mColumns.contents,   csr_columns,    sizeof(int)   * _mNumNonzeroElems );

    memcpy( _mValues.contents,    csr_values,     sizeof(float) * _mNumNonzeroElems );

    memcpy( _mVector.contents,    csr_vector,     sizeof(float) * _mNumColumns );
}


- (float*) getRawPointerOutputVector;
{
    return (float*)_mOutputVector.contents;
}


- (void) performComputation
{
    if ( _mUSeAdaptation ) {

        [ self performComputationAdaptive ];
    }
    else {

        [ self performComputationRowPerThread ];
    }
}


- (void) performComputationRowPerThread
{
    struct sparse_matrix_vector_constants c;
    memset( &c, (uint)0, sizeof(struct sparse_matrix_vector_constants) );
    c.num_rows    = _mNumRows;
    c.num_columns = _mNumColumns;
    c.num_nnz     = _mNumNonzeroElems;
    c.num_blocks  = _mNumBlocks;
    memcpy( _mConst.contents, &c, sizeof(struct sparse_matrix_vector_constants) );

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO_sparse_matrix_vector_row_per_thread ];

    [ computeEncoder setBuffer:_mRowPtrs               offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mColumns               offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mValues                offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mVector                offset:0 atIndex:3 ];
    [ computeEncoder setBuffer:_mOutputVector          offset:0 atIndex:4 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:5 ];

    int numThreadsPerGroup = 1024;
    int numGroupsPerGrid   = (_mNumRows + 1023) / 1024;

    if ( numGroupsPerGrid == 1 && _mNumRows < 1024 ){

        numThreadsPerGroup  = ((_mNumRows + 31) / 32) * 32;
    }
    [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( numThreadsPerGroup, 1, 1) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}


- (void) performComputationAdaptive
{
    struct sparse_matrix_vector_constants c;
    memset( &c, (uint)0, sizeof(struct sparse_matrix_vector_constants) );
    c.num_rows    = _mNumRows;
    c.num_columns = _mNumColumns;
    c.num_nnz     = _mNumNonzeroElems;
    c.num_blocks  = _mNumBlocks;

    memcpy( _mConst.contents, &c, sizeof(struct sparse_matrix_vector_constants) );

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO_sparse_matrix_vector_adaptive ];

    [ computeEncoder setBuffer:_mBlockPtrs             offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mThreadsPerRow         offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mMaxIters              offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mRowPtrs               offset:0 atIndex:3 ];
    [ computeEncoder setBuffer:_mColumns               offset:0 atIndex:4 ];
    [ computeEncoder setBuffer:_mValues                offset:0 atIndex:5 ];
    [ computeEncoder setBuffer:_mVector                offset:0 atIndex:6 ];
    [ computeEncoder setBuffer:_mOutputVector          offset:0 atIndex:7 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:8 ];

    int numThreadsPerGroup = 1024;
    int numGroupsPerGrid   = _mNumBlocks;

    [ computeEncoder dispatchThreadgroups:MTLSizeMake( numGroupsPerGrid,   1, 1 )
                    threadsPerThreadgroup:MTLSizeMake( numThreadsPerGroup, 1, 1 ) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

#import "metal_compute_base.h"

#include <sparse_matrix_vector_metal_cpp.h>

@interface SparseMatrixVectorMetalObjC : MetalComputeBase

- (instancetype) initWithNumRows:(size_t) M
                      NumColumns:(size_t) N
                 NumNonzeroElems:(size_t) num_nonzero_elems
                   UseAdaptation:(bool)   use_adaptation ;

- (float*) getRawPointerOutputVector;

- (void) setInitialStatesBlockPtrs:(int*)   csr_block_ptrs
                     ThreadsPerRow:(int*)   csr_threads_per_row
                          MaxIters:(int*)   csr_max_iters
                         NumBlocks:(size_t) num_blocks
                           RowPtrs:(int*)   csr_row_ptrs
                           Columns:(int*)   csr_columns
                            Values:(float*) csr_values
                            Vector:(float*) csr_vector
                      OutputVector:(float*) output_vector ;

- (void) performComputation;

@end

#import "sparse_matrix_vector_metal_objc.h"
#import "sparse_matrix_vector_metal_cpp_impl.h"

SparseMatrixVectorMetalCppImpl::SparseMatrixVectorMetalCppImpl(
    const int M,
    const int N,
    const int num_nonzero_elems,
    const bool use_adaptation
) {
    m_self = [ [ SparseMatrixVectorMetalObjC alloc ] initWithNumRows: M
                                                          NumColumns: N
                                                     NumNonzeroElems: num_nonzero_elems
                                                       UseAdaptation: use_adaptation     ];
}             

SparseMatrixVectorMetalCppImpl::~SparseMatrixVectorMetalCppImpl(){;}

void SparseMatrixVectorMetalCppImpl::setInitialStates(
    int*      csr_block_ptrs,
    int*      csr_threads_per_row,
    int*      csr_max_iters,
    const int num_blocks,
    int*      csr_row_ptrs,
    int*      csr_columns,
    float*    csr_values,
    float*    csr_vector,
    float*    output_vector
) {
    [ m_self setInitialStatesBlockPtrs: csr_block_ptrs
                         ThreadsPerRow: csr_threads_per_row
                              MaxIters: csr_max_iters
                             NumBlocks: num_blocks
                               RowPtrs: csr_row_ptrs
                               Columns: csr_columns
                                Values: csr_values
                                Vector: csr_vector
                          OutputVector: output_vector ];

}

float* SparseMatrixVectorMetalCppImpl::getRawPointerOutputVector() {

    return [ m_self getRawPointerOutputVector ];
}

void SparseMatrixVectorMetalCppImpl::performComputation()
{
    [ m_self performComputation ];
}



#import "metal_compute_base.h"

#include "dense_matrix_vector_metal_cpp.h"

@interface DenseMatrixVectorMetalObjCOwnShader : MetalComputeBase

- (instancetype) initWithM:(int) m N:(int) n ColMajor:(bool) col_major ThreadsOverRows:(bool) threads_over_rows;

- (void)   setInitialStatesMat:(float*) mat Vec:(float*) v;

- (float*) getRawPointerMat;

- (float*) getRawPointerVec;

- (float*) getRawPointerOutVec;

- (void)   performComputation;

@end

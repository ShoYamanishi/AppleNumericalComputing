#import "metal_compute_base.h"

#include <dense_matrix_vector_metal_cpp.h>

@interface DenseMatrixVectorMetalObjCMPS : MetalComputeBase

- (instancetype) initWithM:(int) m N:(int) n;

- (void) setInitialStatesMat:(float*) mat Vec:(float*) v;

- (float*) getRawPointerMat;

- (float*) getRawPointerVec;

- (float*) getRawPointerOutVec;

- (void) performComputation;

@end

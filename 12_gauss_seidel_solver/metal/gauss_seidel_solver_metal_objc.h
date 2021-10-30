#import "metal_compute_base.h"

#include <gauss_seidel_solver_metal_cpp.h>

@interface GaussSeidelSolverMetalObjC : MetalComputeBase

- (instancetype)initWithDim:(int) dim Iteration:(int) iteration;

- (void) setInitialStatesA:(float*) A D:(float*) D B:(float*) b X1:(float*) x1 X2:(float*) x2;

- (float*) getRawPointerA;

- (float*) getRawPointerB;

- (float*) getRawPointerActiveX;

- (float) getError;

- (void) performComputation;

@end

#import "metal_compute_base.h"

#include "jacobi_solver_metal_cpp.h"

@interface JacobiSolverMetalObjC : MetalComputeBase

- (instancetype)initWithDim:(int) dim Iteration:(int) iteration Type:(int) type OneCommit:(bool) one_commit;

- (void) setInitialStatesA:(float*) A D:(float*) D B:(float*) b X1:(float*) x1 X2:(float*) x2;

- (float*) getRawPointerA;

- (float*) getRawPointerB;

- (float*) getRawPointerActiveX;

- (float) getError;

- (void) performComputation;

@end

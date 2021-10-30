#import "metal_compute_base.h"

#include <cholesky_metal_cpp.h>

@interface CholeskyMetalObjCOwnShader : MetalComputeBase

- (instancetype) initWithDim:(size_t) dim;

- (void) setInitialStatesL:(float*) L B:(float*) b;

- (float*) getRawPointerL;

- (float*) getRawPointerX;

- (float*) getRawPointerY;

- (float*) getRawPointerB;

- (void) performComputation;

@end

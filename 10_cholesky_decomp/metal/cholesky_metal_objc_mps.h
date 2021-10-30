#import "metal_compute_base.h"

@interface CholeskyMetalObjCMPS : MetalComputeBase

- (instancetype) initWithDim:(size_t) dim;

- (void)   setInitialStatesL:(float*) L B:(float*) b;

- (float*) getRawPointerL;

- (float*) getRawPointerX;

- (float*) getRawPointerY;

- (float*) getRawPointerB;

- (void)   performComputation;

@end

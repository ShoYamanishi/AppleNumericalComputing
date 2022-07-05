#import "metal_compute_base.h"

@interface ConjugateGradientMetalObjC : MetalComputeBase

- (instancetype) initWithNumElements: (int)   dim
                  NumThreadsPerGroup: (int)   num_threads_per_group
                    MaxNumIterations: (int)   max_num_iterations
                             Epsilon: (float) epsilon;
- (float*) getRawPointerA;
- (float*) getRawPointerB;
- (void)   performComputation;
- (float*) getX;
- (int)    getNumIterations;
@end

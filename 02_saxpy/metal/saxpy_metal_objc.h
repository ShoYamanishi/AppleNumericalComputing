#import "metal_compute_base.h"

@interface SaxpyMetalObjC : MetalComputeBase
- (instancetype) initWithNumElements:(size_t) numElements 
                  NumThreadsPerGroup:(size_t) numThreadsPerGroup
                    NumGroupsPerGrid:(size_t) numGroupsPerGrid;
- (float*) getRawPointerX;
- (float*) getRawPointerY;
- (void) performComputation;
@property float scalar_a;
@end

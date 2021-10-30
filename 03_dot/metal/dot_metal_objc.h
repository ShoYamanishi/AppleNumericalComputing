#import "metal_compute_base.h"

@interface DotMetalObjC : MetalComputeBase
- (instancetype) initWithNumElements:(size_t) numElements 
                      ReductionType :(int)    recution_type
                  NumThreadsPerGroup:(size_t) num_threads_per_group
                    NumGroupsPerGrid:(size_t) num_groups_per_grid;
- (float*) getRawPointerX;
- (float*) getRawPointerY;
- (void)   performComputation;
- (float)  getDotXY;
@end

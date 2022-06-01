#import "metal_compute_base.h"

@interface BitonicSortMetalObjC : MetalComputeBase
- (instancetype) initWithNumElements:(size_t)  num_elements 
                            forFloat:(bool)    for_float 
           NumThreadsPerThreadgrouop:(size_t)  num_threads_per_threadgroup;

- (uint) numElements;
- (void) performComputation;
- (int*) getRawPointerInOut;
@end

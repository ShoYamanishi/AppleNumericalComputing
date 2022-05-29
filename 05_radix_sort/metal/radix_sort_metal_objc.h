#import "metal_compute_base.h"

@interface RadixSortMetalObjC : MetalComputeBase
- (instancetype) initWithNumElements:(size_t)  num_elements 
                            forFloat:(bool)    for_float 
                      CoalescedWrite:(bool)    coalesced_write 
                            EarlyOut:(bool)    early_out
              NumIterationsPerCommit:(int)     num_iterations_per_commit
           NumThreadsPerThreadgrouop:(size_t)  num_threads_per_threadgroup;
- (void) resetBufferFlag;
- (uint) numElements;
- (int*) getRawPointerIn;
- (int*) getRawPointerOut;
- (void) performComputation;
- (int*) getRawPointerIn1;
- (int*) getRawPointerIn2;
@end

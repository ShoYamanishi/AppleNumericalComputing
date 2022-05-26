#import "metal_compute_base.h"

@interface PrefixSumMetalObjCRecursive : MetalComputeBase

- (instancetype) initWithNumElements:(size_t) numElements
                                Type:(int)    algo_type
                      NumPartialSums:(size_t) num_partial_sums
                            ForFloat:(bool)   for_float
            NumThreadsPerThreadgroup:(int)    num_threads_per_threadgroup;

- (uint)   numElements:(uint) layer;
- (uint)   numThreadsPerGroup:(uint) layer;
- (uint)   numGroupsPerGrid:(uint) layer;
- (int*)   getRawPointerInForInt;
- (float*) getRawPointerInForFloat;
- (int*)   getRawPointerOutForInt;
- (float*) getRawPointerOutForFloat;
- (int*)   getRawPointerGridPrefixSumsForInt:(uint)layer;
- (float*) getRawPointerGridPrefixSumsForFloat:(uint)layer;
- (void)   performComputation;

@end

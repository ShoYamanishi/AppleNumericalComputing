#import "metal_compute_base.h"

@interface PrefixSumMetalObjCMerrillGrimshaw : MetalComputeBase

- (instancetype) initWithNumElements:(size_t) num_elements 
                      NumPartialSums:(size_t) num_partial_sums 
                            ForFloat:(bool)   for_float
            NumThreadsPerThreadgroup:(int)    num_threads_per_threadgroup;
- (uint)   numElements;
- (int*)   getRawPointerInForInt;
- (float*) getRawPointerInForFloat;
- (int*)   getRawPointerOutForInt;
- (float*) getRawPointerOutForFloat;
- (int*)   getRawPointerPartialSumsForInt;
- (float*) getRawPointerPartialSumsForFloat;
- (void)   performComputation;

@end

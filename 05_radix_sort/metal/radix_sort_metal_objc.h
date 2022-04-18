#import "metal_compute_base.h"

@interface RadixSortMetalObjC : MetalComputeBase
- (instancetype) initWithNumElements:(size_t) num_elements forFloat:(bool) for_float CoalescedWrite:(bool) coalesced_write EarlyOut:(bool) early_out InOneCommit:(bool) in_one_commit;
- (void) resetBufferFlag;
- (uint) numElements;
- (int*) getRawPointerIn;
- (int*) getRawPointerOut;
- (void) performComputation;
- (int*) getRawPointerIn1;
- (int*) getRawPointerIn2;
@end

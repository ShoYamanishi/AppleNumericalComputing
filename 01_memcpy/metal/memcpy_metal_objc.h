#import "metal_compute_base.h"

@interface MemcpyMetalObjC : MetalComputeBase
- (instancetype) initWithNumBytes:(size_t) numBytes  UseManagedBuffer:(bool) useManagedBuffer;
- (uint) numBytes;
- (uint) numGroupsPerGrid;
- (uint) numThreadsPerGroup;
- (void*) getRawPointerIn;
- (void*) getRawPointerOut;
- (void) performComputationKernel;
- (void) performComputationBlit;
@end

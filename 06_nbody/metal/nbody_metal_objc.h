#import "metal_compute_base.h"

#include "nbody_metal_cpp.h"

@interface NBodyMetalObjC : MetalComputeBase
- (instancetype) initWithNumElements:(size_t) num_elements ;
- (uint) numElements;
- (struct particle*) getRawPointerParticles;
- (void) performComputationDirectionIsP0ToP1:(bool) p0_to_p1;
@end

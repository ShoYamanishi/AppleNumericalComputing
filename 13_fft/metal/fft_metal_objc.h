#import "metal_compute_base.h"

#include <fft_metal_cpp.h>

@interface FFTMetalObjC : MetalComputeBase

- (instancetype)init;

- (void) setInitialStatesTimeRe:(float*) time_re TimeIm:(float*) time_im;

- (float*) getRawPointerTimeRe;

- (float*) getRawPointerTimeIm;

- (float*) getRawPointerFreqRe;

- (float*) getRawPointerFreqIm;

- (void) performComputation;

@end

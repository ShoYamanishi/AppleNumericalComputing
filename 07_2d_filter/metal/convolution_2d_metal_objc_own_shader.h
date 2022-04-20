#import "metal_compute_base.h"

#include "convolution_2d_metal_cpp.h"

@interface Convolution2DMetalObjCOwnShader : MetalComputeBase

- (instancetype) initWithWidth:(size_t) width Height:(size_t) height KernelSize:(size_t) kernel_size Use2Stages:(bool) use2stages;

- (void)   copyToInputBuffer: (const float* const) p;
- (void)   copyToKernelBuffer:(const float* const) p;
- (float*) getOutputImagePtr;
- (void)   performConvolution;

@end

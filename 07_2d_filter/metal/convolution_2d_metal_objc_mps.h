#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "convolution_2d_metal_cpp.h"

@interface Convolution2DMetalObjCMPS : NSObject

- (instancetype) initWithWidth:(size_t) width Height:(size_t) height KernelSize:(size_t) kernel_size;

- (void)   copyToInputBuffer: (const float* const) p;
- (void)   copyToKernelBuffer:(const float* const) p;
- (float*) getOutputImagePtr;
- (void)   performConvolution;

@end

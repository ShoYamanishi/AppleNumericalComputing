#import <Foundation/Foundation.h>

@interface Convolution2D_CIImageObjc : NSObject
- (instancetype) initWithWidth:(size_t) width Height:(size_t) height KernelSize:(size_t) kernel UseGPU:(bool) use_gpu;
- (void) release_explicit;
- (void) copyToInputBuffer:(const float*)p;
- (void) copyToKernelBuffer:(const float*)p;
- (float*) getOutputImagePtr;
- (void) performConvolution;
@end



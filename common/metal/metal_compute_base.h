#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalComputeBase : NSObject
- (instancetype) init;
- (bool) loadLibraryWithName:(NSString*)name;
- (id<MTLComputePipelineState>) getPipelineStateForFunction:(NSString*)name;
- (id<MTLBuffer>) getSharedMTLBufferForBytes:(int)bytes for:(NSString*)name;
- (id<MTLBuffer>) getPrivateMTLBufferForBytes:(int)bytes for:(NSString*)name;
- (id<MTLBuffer>) getManagedMTLBufferForBytes:(int)bytes for:(NSString*)name;

@property(readonly) id<MTLDevice>       device;
@property(readonly) id<MTLCommandQueue> commandQueue;
@property(readonly) id<MTLLibrary>      library;

@end

#import "metal_compute_base.h"
	
@implementation MetalComputeBase
{
    id<MTLDevice>       _mDevice;
    id<MTLCommandQueue> _mCommandQueue;
    id<MTLLibrary>      _mLibrary;
}

@synthesize device = _mDevice;;
@synthesize commandQueue = _mCommandQueue;
@synthesize library = _mLibrary;

- (instancetype) init
{
    self = [super init];
    if (self)
    {
        _mDevice = MTLCreateSystemDefaultDevice();

        if ( _mDevice == nil ) {
            NSLog(@"Failed to get the default Metal device.");
            return nil;
        }

        _mCommandQueue = [ _mDevice newCommandQueue ];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }
    return self;
}


- (bool) loadLibraryWithName:(NSString*)name
{
    NSError* error = nil;
    _mLibrary = [_mDevice newLibraryWithFile: name error: &error ];
    if ( _mLibrary == nil )
    {
        NSLog(@"Failed to find the default library. Error %@", error );
        return false;
    }
    return true;
}


- (id<MTLComputePipelineState>) getPipelineStateForFunction:(NSString*)name
{
    id<MTLFunction> kernel_func = [ _mLibrary newFunctionWithName: name ];
    if ( kernel_func == nil )
    {
        NSLog( @"Failed to find the function [%@].", name );
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pso = 
        [ _mDevice newComputePipelineStateWithFunction: kernel_func error:&error ];
    if ( pso == nil ) {
        NSLog(@"Failed to created pipeline state object. Error %@.", error);
        return nil;
    }
    return pso;
}


- (id<MTLBuffer>) getSharedMTLBufferForBytes:(int)bytes for:(NSString*)name
{
    id<MTLBuffer> buf = [_mDevice newBufferWithLength: bytes options:MTLResourceStorageModeShared ];
    if ( buf== nil)
    {
        NSLog( @"Failed to allocate new metal buffer [%@].", name );
        return nil;
    }
    return buf;
}


- (id<MTLBuffer>) getPrivateMTLBufferForBytes:(int)bytes for:(NSString*)name
{
    id<MTLBuffer> buf = [_mDevice newBufferWithLength: bytes options:MTLResourceStorageModePrivate ];
    if ( buf== nil)
    {
        NSLog( @"Failed to allocate new metal buffer [%@].", name );
        return nil;
    }
    return buf;
}


- (id<MTLBuffer>) getManagedMTLBufferForBytes:(int)bytes for:(NSString*)name;
{
    id<MTLBuffer> buf = [_mDevice newBufferWithLength: bytes options:MTLResourceStorageModeManaged ];
    if ( buf== nil)
    {
        NSLog( @"Failed to allocate new metal buffer [%@].", name );
        return nil;
    }
    return buf;
}
@end

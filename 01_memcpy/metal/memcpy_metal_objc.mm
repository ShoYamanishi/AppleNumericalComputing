#import <Metal/Metal.h>
#import "memcpy_metal_objc.h"

struct memcpy_constants
{
    uint  num_elements;
};

@implementation MemcpyMetalObjC
{
    id<MTLComputePipelineState> _mPSO;
    id<MTLBuffer>               _mIn;
    id<MTLBuffer>               _mOut;
    id<MTLBuffer>               _mConst;
    uint                        _mNumElementsInt;
    uint                        _mNumGroupsPerGrid;
    uint                        _mNumThreadsPerGroup;
    bool                        _mUseManagedBuffer;
}

- (instancetype) initWithNumBytes:(size_t) num_bytes UseManagedBuffer:(bool) useManagedBuffer
{
    self = [super init];
    if (self)
    {
        _mUseManagedBuffer = useManagedBuffer;

        if (num_bytes < 1024*sizeof(int)) {
            _mNumElementsInt     = (num_bytes + sizeof(int) - 1)/ sizeof(int);
            _mNumThreadsPerGroup = (_mNumElementsInt + 31) / 32 * 32;
            _mNumGroupsPerGrid   = 1;
        }
        else {
            _mNumElementsInt     = (num_bytes + sizeof(int) - 1)/ sizeof(int);
            _mNumThreadsPerGroup = 1024;
            _mNumGroupsPerGrid   = (_mNumElementsInt + 1023) / 1024;
        }

        [ self loadLibraryWithName:@"./memcpy.metallib" ];

        _mPSO = [ self getPipelineStateForFunction:@"my_memcpy" ];



        if ( _mUseManagedBuffer ) {
            _mIn    = [ self getManagedMTLBufferForBytes: _mNumElementsInt * sizeof(int)  for:@"_mIn"  ];
            _mOut   = [ self getManagedMTLBufferForBytes: _mNumElementsInt * sizeof(int)  for:@"_mOut" ];
            _mConst = [ self getManagedMTLBufferForBytes: sizeof(struct memcpy_constants) for:@"_mOut" ];
        }
        else {
            _mIn    = [ self getSharedMTLBufferForBytes: _mNumElementsInt * sizeof(int)  for:@"_mIn"  ];
            _mOut   = [ self getSharedMTLBufferForBytes: _mNumElementsInt * sizeof(int)  for:@"_mOut" ];
            _mConst = [ self getSharedMTLBufferForBytes: sizeof(struct memcpy_constants) for:@"_mOut" ];
        }

        struct memcpy_constants c;
        memset( &c, (uint)0, sizeof(struct memcpy_constants) );
        c.num_elements = _mNumElementsInt;
        memcpy( _mConst.contents, &c, sizeof(struct memcpy_constants) );

    }
    return self;
}

- (uint) numGroupsPerGrid
{
    return _mNumGroupsPerGrid;
}

- (uint) numThreadsPerGroup
{
    return _mNumThreadsPerGroup;
}

- (uint) numBytes
{
    return _mNumElementsInt * sizeof(int);
}


-(void*) getRawPointerIn
{
    return (void*)_mIn.contents;
}

-(void*) getRawPointerOut
{
    return (void*)_mOut.contents;
}

-(void) performComputationKernel
{
    if ( _mUseManagedBuffer ) {

        [_mIn    didModifyRange: NSMakeRange(0, _mNumElementsInt * sizeof(int)   ) ];
        [_mConst didModifyRange: NSMakeRange(0, sizeof( struct memcpy_constants) ) ];
    }

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );


    [ computeEncoder setComputePipelineState: _mPSO ];

    [ computeEncoder setBuffer:_mIn                    offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mOut                   offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:2 ];
    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [ computeEncoder endEncoding];

    if ( _mUseManagedBuffer ) {

        id<MTLBlitCommandEncoder> blitEncoder = [ commandBuffer blitCommandEncoder ];

        assert( blitEncoder != nil );

        [ blitEncoder synchronizeResource:_mOut ];
        [ blitEncoder endEncoding ];
    }

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}


-(void) performComputationBlit
{
    if ( _mUseManagedBuffer ) {

        [_mIn    didModifyRange: NSMakeRange(0, _mNumElementsInt * sizeof(int)   ) ];
    }

    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLBlitCommandEncoder> blitEncoder = [ commandBuffer blitCommandEncoder ];

    assert( blitEncoder != nil );

    [ blitEncoder copyFromBuffer: _mIn
                    sourceOffset: 0
                        toBuffer: _mOut
               destinationOffset: 0 
                            size: sizeof(int)*_mNumElementsInt  ];

    if ( _mUseManagedBuffer ) {

        [ blitEncoder synchronizeResource:_mOut ];
    }

    [blitEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

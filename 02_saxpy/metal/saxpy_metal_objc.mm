#import "saxpy_metal_objc.h"

struct saxpy_constants
{
    uint  num_elements;
};

@implementation SaxpyMetalObjC
{
    id<MTLComputePipelineState> _mPSO;

    id<MTLBuffer> _mX;
    id<MTLBuffer> _mY;
    id<MTLBuffer> _ma;
    id<MTLBuffer> _mConst;

    uint          _mNumElements;
    uint          _mNumThreadsPerGroup;
    uint          _mNumGroupsPerGrid;
}

@synthesize scalar_a ;

- (instancetype) initWithNumElements:(size_t) numElements 
                  NumThreadsPerGroup:(size_t) numThreadsPerGroup
                    NumGroupsPerGrid:(size_t) numGroupsPerGrid
{
    self = [super init];
    if (self)
    {
        _mNumElements        = numElements;
        _mNumThreadsPerGroup = numThreadsPerGroup;
        _mNumGroupsPerGrid   = numGroupsPerGrid;

        [ self loadLibraryWithName:@"./saxpy.metallib" ];

        _mPSO = [ self getPipelineStateForFunction:@"saxpy" ];

        _mX =     [ self getSharedMTLBufferForBytes: _mNumElements * sizeof(float)  for:@"_mX"     ];
        _mY =     [ self getSharedMTLBufferForBytes: _mNumElements * sizeof(float)  for:@"_mY"     ];
        _ma =     [ self getSharedMTLBufferForBytes: sizeof(float)                  for:@"_ma"     ];
        _mConst = [ self getSharedMTLBufferForBytes: sizeof(struct saxpy_constants) for:@"_mConst" ];

        struct saxpy_constants c;
        memset( &c, (int)0, sizeof(struct saxpy_constants) );
        c.num_elements = _mNumElements;
        memcpy( _mConst.contents, &c, sizeof(struct saxpy_constants) );
    }
    return self;
}

-(float*) getRawPointerX
{
    return (float*)_mX.contents;
}

-(float*) getRawPointerY
{
    return (float*)_mY.contents;
}

-(void) performComputation
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState:_mPSO ];
    [ computeEncoder setBuffer:_mX     offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mY     offset:0 atIndex:1 ];
    ((float*)_ma.contents)[0] = self.scalar_a;
    [ computeEncoder setBuffer:_ma     offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mConst offset:0 atIndex:3 ];

    [computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1 )
                   threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1 ) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

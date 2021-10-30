#import "nbody_metal_objc.h"

@implementation NBodyMetalObjC
{
    id<MTLComputePipelineState> _mPSO_nbody_naive_p0_to_p1;
    id<MTLComputePipelineState> _mPSO_nbody_naive_p1_to_p0;
    id<MTLBuffer>               _mParticles;
    id<MTLBuffer>               _mConst;

    uint                        _mNumElements;
    uint                        _mNumGroupsPerGrid;
    uint                        _mNumThreadsPerGroup;
}

- (instancetype) initWithNumElements:(size_t) num_elements 
{
    self = [super init];
    if (self)
    {
        _mNumElements        = num_elements;
        if ( _mNumElements <= 1024 ) {
            _mNumThreadsPerGroup = ((num_elements + 31)/32) * 32;
            _mNumGroupsPerGrid   = 1;
        }
        else {
            _mNumThreadsPerGroup = 1024;
            _mNumGroupsPerGrid   = (num_elements + 1023) / 1024;
        }

        [ self loadLibraryWithName:@"./nbody.metallib" ];

        _mPSO_nbody_naive_p0_to_p1 = [ self getPipelineStateForFunction:@"nbody_naive_p0_to_p1" ];
        _mPSO_nbody_naive_p1_to_p0 = [ self getPipelineStateForFunction:@"nbody_naive_p1_to_p0" ];
       
        _mParticles = [ self getSharedMTLBufferForBytes: _mNumElements * sizeof(struct particle) for:@"_mParticles" ];
        _mConst     = [ self getSharedMTLBufferForBytes: sizeof(struct nbody_constants)          for:@"_mConst" ];
        struct nbody_constants c;
        memset( &c, (uint)0, sizeof(struct nbody_constants) );
        c.num_elements = num_elements;
        c.EPSILON = 1.0e-10;
        c.COEFF_G = 9.8;
        c.delta_t = 0.1;
        memcpy( _mConst.contents, &c, sizeof(struct nbody_constants) );
    }
    return self;
}


- (uint) numElements
{
    return _mNumElements;
}


-(struct particle*) getRawPointerParticles
{
    return (struct particle*)_mParticles.contents;
}


-(void) performComputationDirectionIsP0ToP1:(bool) p0_to_p1
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    if (p0_to_p1) {
        [ computeEncoder setComputePipelineState: _mPSO_nbody_naive_p0_to_p1 ];
    }
    else {
        [ computeEncoder setComputePipelineState: _mPSO_nbody_naive_p1_to_p0 ];
    }

    [ computeEncoder setBuffer:_mParticles             offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mConst                 offset:0 atIndex:1 ];
    [ computeEncoder dispatchThreadgroups:MTLSizeMake( _mNumGroupsPerGrid,   1, 1)
                    threadsPerThreadgroup:MTLSizeMake( _mNumThreadsPerGroup, 1, 1) ];

    [ computeEncoder endEncoding ];

    [ commandBuffer commit ];

    [ commandBuffer waitUntilCompleted ];
}

@end

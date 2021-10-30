#include <memory>
#include <algorithm>
#include <iostream>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "dense_matrix_vector_metal_objc_mps.h"

@implementation DenseMatrixVectorMetalObjCMPS
{
    int _mM;
    int _mN;

    MPSMatrixVectorMultiplication*  _mMPSMult;
    id<MTLBuffer>                   _mMPSMatBuffer;
    id<MTLBuffer>                   _mMPSVecBuffer;
    id<MTLBuffer>                   _mMPSOutVecBuffer;

    MPSMatrix*                      _mMPSMat;
    MPSVector*                      _mMPSVec;
    MPSVector*                      _mMPSOutVec;
}

- (instancetype) initWithM:(int) m N:(int) n
{

    self = [super init];

    if (self)
    {
        _mM    = m;
        _mN    = n;

        _mMPSMult = [ [ MPSMatrixVectorMultiplication alloc] initWithDevice: self.device rows: _mM columns: _mN ];
        if (_mMPSMult == nil)
        {
            NSLog(@"Failed to allocate MPSMult.");
            return nil;
        }

        _mMPSMatBuffer    = [ self getSharedMTLBufferForBytes: sizeof(float)* _mM *_mN for:@"_mMPSMatBuffer"    ];
        _mMPSVecBuffer    = [ self getSharedMTLBufferForBytes: sizeof(float)* _mN      for:@"_mMPSVecBuffer"    ];
        _mMPSOutVecBuffer = [ self getSharedMTLBufferForBytes: sizeof(float)* _mM      for:@"_mMPSOutVecBuffer" ];

        MPSMatrixDescriptor* desc_mat
            = [ MPSMatrixDescriptor matrixDescriptorWithRows: _mM
                                                     columns: _mN
                                                    rowBytes: sizeof(float)*_mN
                                                    dataType: MPSDataTypeFloat32   ];
        if ( desc_mat == nil)
        {
            NSLog(@"Failed to allocate MPSMatrixDescriptor");
            return nil;
        }

        _mMPSMat = [ [ MPSMatrix alloc ] initWithBuffer: _mMPSMatBuffer descriptor: desc_mat ];
        if ( _mMPSMat == nil )
        {
            NSLog(@"Failed to allocate _mMPSMat.");
            return nil;
        }

        MPSVectorDescriptor* desc_vec
            = [ MPSVectorDescriptor vectorDescriptorWithLength:_mN dataType: MPSDataTypeFloat32 ];
        if ( desc_vec == nil)
        {
            NSLog(@"Failed to allocate MPSVectorDescriptor ckp1");
            return nil;
        }

        _mMPSVec = [ [ MPSVector alloc ] initWithBuffer:_mMPSVecBuffer descriptor: desc_vec ];
        if ( _mMPSVec == nil )
        {
            NSLog(@"Failed to allocate _mMPSVec.");
            return nil;
        }

        MPSVectorDescriptor* desc_outvec
            = [ MPSVectorDescriptor vectorDescriptorWithLength:_mM dataType: MPSDataTypeFloat32 ];
        if ( desc_outvec == nil)
        {
            NSLog(@"Failed to allocate MPSVectorDescriptor ckp2");
            return nil;
        }

        _mMPSOutVec = [ [ MPSVector alloc ] initWithBuffer:_mMPSOutVecBuffer descriptor: desc_outvec ];
        if ( _mMPSOutVec == nil )
        {
            NSLog(@"Failed to allocate _mMPSOutVec.");
            return nil;
        }

    }
    return self;
}


- (void) setInitialStatesMat:(float*) mat Vec:(float*) v;
{
    memcpy( _mMPSMatBuffer.contents, mat, _mM * _mN * sizeof(float) );
    memcpy( _mMPSVecBuffer.contents, v,   _mN * sizeof(float) );
}


- (float*) getRawPointerOutVec
{
    return (float*)_mMPSOutVecBuffer.contents;
}


- (float*) getRawPointerVec
{
    return (float*)_mMPSVecBuffer.contents;
}


- (float*) getRawPointerMat
{
    return (float*)_mMPSMatBuffer.contents;
}


- (void) performComputation
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    [ _mMPSMult encodeToCommandBuffer:  commandBuffer 
                          inputMatrix:  _mMPSMat
                          inputVector:  _mMPSVec
                          resultVector: _mMPSOutVec ];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

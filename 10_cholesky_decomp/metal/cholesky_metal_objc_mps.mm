#include <memory>
#include <algorithm>
#include <iostream>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "cholesky_metal_objc_mps.h"


@implementation CholeskyMetalObjCMPS
{
    uint                            _mDim;

    MPSMatrixDecompositionCholesky* _mMPSDecomposer;
    id<MTLBuffer>                   _mMPSMatrixBufferIn;
    id<MTLBuffer>                   _mMPSMatrixBufferOut;
    id<MTLBuffer>                   _mMPSStatus;
    MPSMatrix*                      _mMPSMatrixIn;
    MPSMatrix*                      _mMPSMatrixOut;

    MPSMatrixSolveTriangular*       _mMPSSolverLyeb;
    MPSMatrixSolveTriangular*       _mMPSSolverLtxey;
    id<MTLBuffer>                   _mMPSBufferx;
    id<MTLBuffer>                   _mMPSBuffery;
    id<MTLBuffer>                   _mMPSBufferb;
    MPSMatrix*                      _mMPSx;
    MPSMatrix*                      _mMPSy;
    MPSMatrix*                      _mMPSb;
}


- (instancetype) initWithDim:(size_t) dim
{
    self = [super init];

    if (self)
    {
        _mDim = dim;

        _mMPSDecomposer = [ [ MPSMatrixDecompositionCholesky alloc] initWithDevice: self.device lower:true order: _mDim ];
        if (_mMPSDecomposer == nil)
        {
            NSLog(@"Failed to allocate MPSDecomposer.");
            return nil;
        }

        _mMPSStatus          = [ self getSharedMTLBufferForBytes: sizeof(int)                    for: @"_mMPSStatus"          ];
        _mMPSMatrixBufferIn  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim  * _mDim for: @"_mMPSMatrixBufferIn"  ];
        _mMPSMatrixBufferOut = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim  * _mDim for: @"_mMPSMatrixBufferOut" ];

        MPSMatrixDescriptor* desc
            = [ MPSMatrixDescriptor matrixDescriptorWithRows: _mDim
                                                     columns: _mDim
                                                    rowBytes: sizeof(float)*_mDim
                                                    dataType: MPSDataTypeFloat32   ];
        if ( desc == nil)
        {
            NSLog(@"Failed to allocate MPSMatrixDescriptor");
            return nil;
        }

        _mMPSMatrixIn = [ [ MPSMatrix alloc ] initWithBuffer: _mMPSMatrixBufferIn
                                                  descriptor: desc ];
        if ( _mMPSMatrixIn == nil )
        {
            NSLog(@"Failed to allocate _mMPSMatrixIn.");
            return nil;
        }

        _mMPSMatrixOut= [ [ MPSMatrix alloc ] initWithBuffer: _mMPSMatrixBufferOut
                                                  descriptor: desc ];
        if ( _mMPSMatrixOut == nil )
        {
            NSLog(@"Failed to allocate _mMPSMatrixOut.");
            return nil;
        }

        _mMPSSolverLyeb = [ [MPSMatrixSolveTriangular alloc]
                            initWithDevice: self.device
                                     right: false
                                     upper: false
                                 transpose: false
                                      unit: false
                                     order: _mDim
                    numberOfRightHandSides: 1
                                     alpha: 1.0  ];
        if ( _mMPSSolverLyeb == nil )
        {
            NSLog(@"Failed to allocate _mMPSSolverLyeb.");
            return nil;
        }

        _mMPSSolverLtxey = [ [MPSMatrixSolveTriangular alloc]
                            initWithDevice: self.device
                                     right: false
                                     upper: false
                                 transpose: true
                                      unit: false
                                     order: _mDim
                    numberOfRightHandSides: 1
                                     alpha: 1.0  ];
        if ( _mMPSSolverLtxey == nil )
        {
            NSLog(@"Failed to allocate _mMPSSolverLtxey.");
            return nil;
        }

        _mMPSBufferx  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim  for: @"_mMPSBufferx" ];
        _mMPSBuffery  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim  for: @"_mMPSBuffery" ];
        _mMPSBufferb  = [ self getSharedMTLBufferForBytes: sizeof(float) * _mDim  for: @"_mMPSBufferb" ];

        MPSMatrixDescriptor* desc_vec
            = [ MPSMatrixDescriptor matrixDescriptorWithRows: _mDim
                                                     columns: 1
                                                    rowBytes: sizeof(float)
                                                    dataType: MPSDataTypeFloat32   ];
        if ( desc_vec == nil)
        {
            NSLog(@"Failed to allocate MPSMatrixDescriptor for vector");
            return nil;
        }

        _mMPSx = [ [ MPSMatrix alloc ] initWithBuffer:_mMPSBufferx descriptor: desc_vec ];
        if ( _mMPSx == nil )
        {
            NSLog(@"Failed to allocate _mMPSx.");
            return nil;
        }

        _mMPSy = [ [ MPSMatrix alloc ] initWithBuffer:_mMPSBuffery descriptor: desc_vec ];
        if ( _mMPSy == nil )
        {
            NSLog(@"Failed to allocate _mMPSy.");
            return nil;
        }

        _mMPSb = [ [ MPSMatrix alloc ] initWithBuffer:_mMPSBufferb descriptor: desc_vec ];
        if ( _mMPSb == nil )
        {
            NSLog(@"Failed to allocate _mMPSb.");
            return nil;
        }
    }
    return self;
}


static inline int lower_mat_index( const int i, const int j, const int dim )
{
    const int num_elems = (dim + 1) * dim / 2;
    const int i_rev = (dim -1) - i;
    const int j_rev = (dim -1) - j;
    return num_elems - 1 - ( j_rev * (j_rev + 1) /2 + i_rev );
}


- (void) setInitialStatesL:(float*) L B:(float*) b
{
    for ( int i = 0; i < _mDim; i++ ) {

        for ( int j = 0 ;j <= i; j++ ) {
            ((float*)_mMPSMatrixBufferIn.contents)[ i * _mDim + j ]  = L[ lower_mat_index( i, j, _mDim ) ];
            ((float*)_mMPSMatrixBufferIn.contents)[ j * _mDim + i ]  = L[ lower_mat_index( i, j, _mDim ) ];
        }
    }

    memcpy( _mMPSBufferb.contents, b, _mDim * sizeof(float) );

}


- (float*) getRawPointerL
{
    return (float*)_mMPSMatrixBufferOut.contents;
}


- (float*) getRawPointerX
{
    return (float*)_mMPSBufferx.contents;
}


- (float*) getRawPointerY
{
    return (float*)_mMPSBuffery.contents;
}


- (float*) getRawPointerB
{
    return (float*)_mMPSBufferb.contents;
}


- (void) performComputation
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );
    
    [ _mMPSDecomposer encodeToCommandBuffer: commandBuffer 
                               sourceMatrix: _mMPSMatrixIn
                               resultMatrix: _mMPSMatrixOut
                                     status: _mMPSStatus    ];

    [ _mMPSSolverLyeb encodeToCommandBuffer: commandBuffer 
                               sourceMatrix: _mMPSMatrixOut
                        rightHandSideMatrix: _mMPSb
                             solutionMatrix: _mMPSy         ];

    [ _mMPSSolverLtxey encodeToCommandBuffer: commandBuffer 
                                sourceMatrix: _mMPSMatrixOut
                         rightHandSideMatrix: _mMPSy
                              solutionMatrix: _mMPSx        ];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];

    MPSMatrixDecompositionStatus status;
    status = (MPSMatrixDecompositionStatus)((int*)_mMPSStatus.contents)[0];
    if ( status != MPSMatrixDecompositionStatusSuccess ) {
        std::cerr << "Error: MPSStatus: " << status << "\n";
    }
}

@end

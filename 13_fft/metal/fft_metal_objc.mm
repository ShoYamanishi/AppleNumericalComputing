#include <memory>
#include <algorithm>
#include <iostream>

#import "fft_metal_objc.h"

@implementation FFTMetalObjC
{
    id<MTLComputePipelineState> _mPSO;

    id<MTLBuffer>               _mTimeRe;
    id<MTLBuffer>               _mTimeIm;
    id<MTLBuffer>               _mFreqRe;
    id<MTLBuffer>               _mFreqIm;
    id<MTLBuffer>               _mShuffledIndices;
    id<MTLBuffer>               _mCosTable;
    id<MTLBuffer>               _mSinTable;
}

- (instancetype)init
{
    self = [super init];

    if (self)
    {

        [ self loadLibraryWithName: @"./fft.metallib" ];

        _mPSO =  [ self getPipelineStateForFunction: @"fft_512_radix_2" ];

        _mTimeRe = [ self getSharedMTLBufferForBytes: sizeof(float) * 512 for: @"_mTimeRe" ];
        _mTimeIm = [ self getSharedMTLBufferForBytes: sizeof(float) * 512 for: @"_mTimeIm" ];
        _mFreqRe = [ self getSharedMTLBufferForBytes: sizeof(float) * 512 for: @"_mFreqRe" ];
        _mFreqIm = [ self getSharedMTLBufferForBytes: sizeof(float) * 512 for: @"_mFreqIm" ];
        _mShuffledIndices = [ self getSharedMTLBufferForBytes: sizeof(int) * 512 for: @"_mShuffledIndices" ];

        _mCosTable = [ self getSharedMTLBufferForBytes: sizeof(float) * 256 for: @"_mCosTable" ];
        _mSinTable = [ self getSharedMTLBufferForBytes: sizeof(float) * 256 for: @"_mSinTable" ];

        [ self prepare_shuffled_indices ];

        [ self prepare_trig_tables ];
    }
    return self;
}

- (void) prepare_trig_tables
{
    float* cos_table = (float*)_mCosTable.contents;
    float* sin_table = (float*)_mSinTable.contents;

    for ( int k = 0; k < 256 ; k++ ) {

        const float theta = -1.0 * M_PI * (float)k / 256.0;

        cos_table[ k ] = cos( theta ); // re
        sin_table[ k ] = sin( theta ); // im
    }
}

void deinterleave( int* in, const int len, int* out )
{
    int* out_even = out;
    int* out_odd  = &(out[len/2]);

    for ( int i = 0; i < len; i++ )  {
        if ( i % 2 == 0 ) {
            out_even[i/2] = in[i];
        }
        else {
            out_odd[i/2]  = in[i];
        }
    }
    if (len > 4) {
        deinterleave( out_even, len/2, in           );
        deinterleave( out_odd,  len/2, &(in[len/2]) );
    }    
}

- (void) prepare_shuffled_indices
{
    int* dst = (int*)_mShuffledIndices.contents;

    int indices1[ 512 ];
    int indices2[ 512 ];

    for ( int i = 0; i < 512; i++ ) {
        indices1[i] = i;
    }

    deinterleave( indices1, 512, indices2 );

    for ( int i = 0; i < 512; i++ ) {
        dst[i] = indices1[i];
    }
}

- (void) setInitialStatesTimeRe:(float*) time_re TimeIm:(float*) time_im
{
    memcpy( _mTimeRe.contents, time_re, 512 * sizeof(float) );
    memcpy( _mTimeIm.contents, time_im, 512 * sizeof(float) );
}

- (float*) getRawPointerTimeRe
{
    return (float*)_mTimeRe.contents;
}

- (float*) getRawPointerTimeIm
{
    return (float*)_mTimeIm.contents;
}

- (float*) getRawPointerFreqRe
{
    return (float*)_mFreqRe.contents;
}

- (float*) getRawPointerFreqIm
{
    return (float*)_mFreqIm.contents;
}

- (void) performComputation
{
    id<MTLCommandBuffer> commandBuffer = [ self.commandQueue commandBuffer ];

    assert( commandBuffer != nil );

    id<MTLComputeCommandEncoder> computeEncoder = [ commandBuffer computeCommandEncoder ];

    assert( computeEncoder != nil );

    [ computeEncoder setComputePipelineState: _mPSO ];

    [ computeEncoder setBuffer:_mTimeRe            offset:0 atIndex:0 ];
    [ computeEncoder setBuffer:_mTimeIm            offset:0 atIndex:1 ];
    [ computeEncoder setBuffer:_mFreqRe            offset:0 atIndex:2 ];
    [ computeEncoder setBuffer:_mFreqIm            offset:0 atIndex:3 ];
    [ computeEncoder setBuffer:_mShuffledIndices   offset:0 atIndex:4 ];
    [ computeEncoder setBuffer:_mCosTable          offset:0 atIndex:5 ];
    [ computeEncoder setBuffer:_mSinTable          offset:0 atIndex:6 ];

    [ computeEncoder dispatchThreadgroups:MTLSizeMake(   1,   1,   1 )
                    threadsPerThreadgroup:MTLSizeMake( 512,   1,   1 ) ];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

@end

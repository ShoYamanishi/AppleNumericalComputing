#include <metal_stdlib>

using namespace metal;

kernel void fft_512_radix_2 (

    device       float*              time_re                        [[ buffer(0) ]],

    device       float*              time_im                        [[ buffer(1) ]],

    device       float*              freq_re                        [[ buffer(2) ]],

    device       float*              freq_im                        [[ buffer(3) ]],

    device const int*                shuffled_indices               [[ buffer(4) ]],

    device const float*              cos_table                      [[ buffer(5) ]],

    device const float*              sin_table                      [[ buffer(6) ]],

    const        uint                tid                            [[ thread_position_in_threadgroup ]]
) {

    threadgroup float v_re[512];
    threadgroup float v_im[512];

    // preparation

    threadgroup float cos_cache[256];
    threadgroup float sin_cache[256];

    if ( tid < 256 ) {
        cos_cache[ tid ] = cos_table[ tid ];
    }
    else {
        sin_cache[ tid - 256 ] = sin_table[ tid - 256 ];
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    float re_this = time_re[ shuffled_indices[ tid ] ];
    float im_this = time_im[ shuffled_indices[ tid ] ];

    float tw_re;
    float tw_im;
    float offset_re;
    float offset_im;

    // straddle 1

    float re_other  = simd_shuffle_xor( re_this, 0x1 );
    float im_other  = simd_shuffle_xor( im_this, 0x1 );
    int   lane = tid % 2;
    if (lane == 0){
        re_this = re_this + re_other;
        im_this = im_this + im_other;
    }
    else {
        re_this = re_other - re_this;
        im_this = im_other - im_this;
    }

    // straddle 2

    re_other = simd_shuffle_xor( re_this, 0x2 );
    im_other = simd_shuffle_xor( im_this, 0x2 );
    lane = tid % 4;

    tw_re = cos_cache[ 0xff & (128 * lane) ];
    tw_im = sin_cache[ 0xff & (128 * lane) ];

    if ( lane < 2 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    // straddle 4

    re_other = simd_shuffle_xor( re_this, 0x4 );
    im_other = simd_shuffle_xor( im_this, 0x4 );
    lane = tid % 8;

    tw_re = cos_cache[ 0xff & (64 * lane) ];
    tw_im = sin_cache[ 0xff & (64 * lane) ];

    if ( lane < 4 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    // straddle 8

    re_other  = simd_shuffle_xor( re_this, 0x8 );
    im_other  = simd_shuffle_xor( im_this, 0x8 );
    lane = tid % 16;

    tw_re     = cos_cache[ 0xff & (32 * lane) ];
    tw_im     = sin_cache[ 0xff & (32 * lane) ];

    if ( lane < 8 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    // straddle 16

    re_other  = simd_shuffle_xor( re_this, 0x10 );
    im_other  = simd_shuffle_xor( im_this, 0x10 );
    lane = tid % 32;

    tw_re     = cos_cache[ 0xff & (16 * lane) ];
    tw_im     = sin_cache[ 0xff & (16 * lane) ];

    if ( lane < 16 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    v_re[ tid ] = re_this;
    v_im[ tid ] = im_this;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // straddle 32

    lane = tid % 64;
    re_other  = (lane<32) ? v_re[tid + 32] : v_re[tid - 32];
    im_other  = (lane<32) ? v_im[tid + 32] : v_im[tid - 32];

    tw_re     = cos_cache[ 0xff & (8 * lane) ];
    tw_im     = sin_cache[ 0xff & (8 * lane) ];

    if ( lane < 32 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    v_re[ tid ] = re_this;
    v_im[ tid ] = im_this;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // straddle 64

    lane = tid % 128;
    re_other  = (lane<64) ? v_re[tid + 64] : v_re[tid - 64];
    im_other  = (lane<64) ? v_im[tid + 64] : v_im[tid - 64];

    tw_re     = cos_cache[ 0xff & (4 * lane) ];
    tw_im     = sin_cache[ 0xff & (4 * lane) ];

    if ( lane < 64 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    v_re[ tid ] = re_this;
    v_im[ tid ] = im_this;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // straddle 128

    lane = tid % 256;
    re_other  = (lane<128) ? v_re[tid + 128] : v_re[tid - 128];
    im_other  = (lane<128) ? v_im[tid + 128] : v_im[tid - 128];

    tw_re     = cos_cache[ 0xff & (2 * lane) ];
    tw_im     = sin_cache[ 0xff & (2 * lane) ];

    if ( lane < 128 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    v_re[ tid ] = re_this;
    v_im[ tid ] = im_this;

    threadgroup_barrier( mem_flags::mem_threadgroup );

    // straddle 256

    lane = tid % 512;
    re_other  = (lane<256) ? v_re[tid + 256] : v_re[tid - 256];
    im_other  = (lane<256) ? v_im[tid + 256] : v_im[tid - 256];

    tw_re     = cos_cache[ 0xff & lane ];
    tw_im     = sin_cache[ 0xff & lane ];

    if ( lane < 256 ) {
        offset_re = tw_re * re_other - tw_im * im_other;
        offset_im = tw_re * im_other + tw_im * re_other;

        re_this = re_this + offset_re;
        im_this = im_this + offset_im;
    }
    else {
        offset_re = tw_re * re_this - tw_im * im_this;
        offset_im = tw_re * im_this + tw_im * re_this;

        re_this = re_other - offset_re;
        im_this = im_other - offset_im;
    }

    freq_re[ tid ] = re_this;
    freq_im[ tid ] = im_this;
}



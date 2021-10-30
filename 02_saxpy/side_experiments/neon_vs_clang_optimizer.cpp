// clang++ -O3 -Wall -pthread -march=armv8-a+fp+simd -std=c++17 -S -c -o tmp.S neon_vs_clang_optimizer.cpp

#include <type_traits>
#include "arm_neon.h"

void saxpy_baseline_float( const float* const x, float* const y, const float a, const size_t num )
{
    for ( size_t i = 0; i < num; i++ ) {
        y[i] = a * x[i] + y[i];
    }
}


void saxpy_baseline_double( const double* const x, double* const y, const double a, const size_t num )
{
    for ( size_t i = 0; i < num; i++ ) {
        y[i] = a * x[i] + y[i];
    }
}


void saxpy_neon_float( const float* const x, float* const y, const float a, const size_t num )
{
    for (size_t i = 0;  i < num; i += 4 ) {

        const float32x4_t x_quad    = vld1q_f32( &x[i] );
        const float32x4_t ax_quad   = vmulq_n_f32( x_quad, a );
        const float32x4_t y_quad    = vld1q_f32( &y[i] );
        const float32x4_t axpy_quad = vaddq_f32( ax_quad, y_quad );
        vst1q_f32( &(y[i]), axpy_quad );
    }
}

void saxpy_neon_float_loop_unrolling_2( const float* const x, float* const y, const float a, const size_t num )
{
    for (size_t i = 0;  i < num; i += 8 ) {

        const float32x4_t x_quad1    = vld1q_f32( &x[i] );
        const float32x4_t x_quad2    = vld1q_f32( &x[i+4] );
        const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, a );
        const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, a );
        const float32x4_t y_quad1    = vld1q_f32( &y[i] );
        const float32x4_t y_quad2    = vld1q_f32( &y[i+4] );
        const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
        const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
        vst1q_f32( &(y[i]), axpy_quad1 );
        vst1q_f32( &(y[i+4]), axpy_quad2 );
    }
}

void saxpy_neon_float_loop_unrolling_4( const float* const x, float* const y, const float a, const size_t num )
{
    for (size_t i = 0;  i < num; i += 16 ) {

        const float32x4_t x_quad1    = vld1q_f32( &x[i] );
        const float32x4_t x_quad2    = vld1q_f32( &x[i+4] );
        const float32x4_t x_quad3    = vld1q_f32( &x[i+8] );
        const float32x4_t x_quad4    = vld1q_f32( &x[i+16] );
        const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, a );
        const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, a );
        const float32x4_t ax_quad3   = vmulq_n_f32( x_quad3, a );
        const float32x4_t ax_quad4   = vmulq_n_f32( x_quad4, a );
        const float32x4_t y_quad1    = vld1q_f32( &y[i] );
        const float32x4_t y_quad2    = vld1q_f32( &y[i+4] );
        const float32x4_t y_quad3    = vld1q_f32( &y[i+8] );
        const float32x4_t y_quad4    = vld1q_f32( &y[i+16] );
        const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
        const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
        const float32x4_t axpy_quad3 = vaddq_f32( ax_quad3, y_quad3 );
        const float32x4_t axpy_quad4 = vaddq_f32( ax_quad4, y_quad4 );
        vst1q_f32( &(y[i]), axpy_quad1 );
        vst1q_f32( &(y[i+4]), axpy_quad2 );
        vst1q_f32( &(y[i+8]), axpy_quad3 );
        vst1q_f32( &(y[i+16]), axpy_quad4 );
    }
}

void saxpy_neon_double( const double* const x, double* const y, const double a, const size_t num )
{
    for (size_t i = 0;  i < num; i += 2 ) {

        const float64x2_t x_quad    = vld1q_f64( &x[i] );
        const float64x2_t ax_quad   = vmulq_n_f64( x_quad, a );
        const float64x2_t y_quad    = vld1q_f64( &y[i] );
        const float64x2_t axpy_quad = vaddq_f64( ax_quad, y_quad );
        vst1q_f64( &(y[i]), axpy_quad );
    }
}


void saxpy_neon_double_loop_unrolling_2( const double* const x, double* const y, const double a, const size_t num )
{
    for (size_t i = 0;  i < num; i += 4 ) {

        const float64x2_t x_quad1    = vld1q_f64( &x[i] );
        const float64x2_t x_quad2    = vld1q_f64( &x[i+4] );
        const float64x2_t ax_quad1   = vmulq_n_f64( x_quad1, a );
        const float64x2_t ax_quad2   = vmulq_n_f64( x_quad2, a );
        const float64x2_t y_quad1    = vld1q_f64( &y[i] );
        const float64x2_t y_quad2    = vld1q_f64( &y[i+4] );
        const float64x2_t axpy_quad1 = vaddq_f64( ax_quad1, y_quad1 );
        const float64x2_t axpy_quad2 = vaddq_f64( ax_quad2, y_quad2 );
        vst1q_f64( &(y[i]), axpy_quad1 );
        vst1q_f64( &(y[i+4]), axpy_quad2 );
    }
}


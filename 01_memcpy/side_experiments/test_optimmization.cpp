// clang++ -O3 -Wall -pthread -march=armv8-a+fp+simd -std=c++17 -S -c -o tmp.S test_optimization.cpp
#include <memory>
#include <type_traits>
#include "arm_neon.h"


void copy_baseline(
//    const int* const __attribute__((aligned(64))) x,
//    int* const       __attribute__((aligned(64))) y,
    const int* const x
    int* const       y
    const size_t num )
{
    for ( size_t i = 0; i < num; i++ ) {
        y[i] = x[i];
    }
}
/*
void copy_baseline_unroll_2( const int* const x, int* const y, const size_t num )
{
    for ( size_t i = 0; i < num; i+=2 ) {
        y[i  ] = x[i  ];
        y[i+1] = x[i+1];
    }
}

void copy_baseline_unroll_4( const int* const x, int* const y, const size_t num )
{
    for ( size_t i = 0; i < num; i+=4 ) {
        y[i  ] = x[i  ];
        y[i+1] = x[i+1];
        y[i+2] = x[i+2];
        y[i+3] = x[i+3];
    }
}

void copy_baseline_unroll_8( const int* const x, int* const y, const size_t num )
{
    for ( size_t i = 0; i < num; i+=8 ) {
        y[i  ] = x[i  ];
        y[i+1] = x[i+1];
        y[i+2] = x[i+2];
        y[i+3] = x[i+3];
        y[i+4] = x[i+4];
        y[i+5] = x[i+5];
        y[i+6] = x[i+6];
        y[i+7] = x[i+7];
    }
}

void copy_memcpy( const int* const x, int* const y, const size_t num )
{
    memcpy(y, x, sizeof(int)*num);

}

void copy_memcpy_32( const int* const x, int* const y )
{
    memcpy(y, x, sizeof(int)*32);
}

void copy_memcpy_64( const int* const x, int* const y )
{
    memcpy(y, x, sizeof(int)*64);
}

void copy_memcpy_128( const int* const x, int* const y )
{
    memcpy(y, x, sizeof(int)*128);
}
*/

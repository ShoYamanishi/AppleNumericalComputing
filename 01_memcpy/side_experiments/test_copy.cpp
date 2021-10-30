// clang++ -O3 -Wall -pthread -march=armv8-a+fp+simd -std=c++17 -S -c -o test_copy.S test_copy.cpp

void copy_baseline( const int* const x, int* const y, const int num )
{
    for ( int i = 0; i < num; i++ ) {
        y[i] = x[i];
    }
}

void copy_baseline2(
    const int* const __attribute__((aligned(64))) x,
    int* const       __attribute__((aligned(64))) y,
    const int num
) {
    # pragma unroll 8
    for ( int i = 0; i < num; i++ ) {
        y[i] = x[i];
    }
}

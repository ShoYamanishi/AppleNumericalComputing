# 512-Point Radix 2 FFT


# 1. Key Points

* Implementations made in Plain C++, NEON, and Metal.

* The algorithm optimized as much as possible with the pre-arranged shuffle indices, and the *twiddle factor* (cos and sin) tables.

* vDSP shows the best performance as expected under 1 micro second.

* The NEON implementation performs approximately 50% slower than vDSP for *float* and 100% slower for *double*.

# 2. Background and Context
FFT is used mainly in the real time audio processing.
It is well known that vDSP provides the highly optimized routine.

# 3. Purpose
The purpose is to measure the relative superiority of vDSP compared with other implementations.
Another purpose it to measure the performance gain of using NEON and multithreads.
Also it is implemented in a Metal kernel to study the behavior.

# 4. Results on Running Time

## 4.1. Overview, Float

| Type          | Time in microseconds | Description                                     |
| ------------- | --------------------:| ----------------------------------------------- |
| CPP_BLOCK 1 1 |    1.69              | Plain C++ implementation - baseline             |
| NEON 1 1      |    1.20              | NEON Intrinsics                                 |
| NEON 1 4      |   21.1               | NEON Intrinsics with 4 threads                  |
| METAL 0 0     |  147.0               | METAL kernel                                    |
| VDSP  1 1     |    0.832             | vDSP_DFT_zop_CreateSetup() & vDSP_DFT_Execute() |

### Remarks
* VDSP runs in sub-micro second. NEON 1 1 version runs about 50% slower than VDSP.
* The overhead of multi-threading is too big and not worth it.
* The overhead of launching a Metal kernel is too big and not worth it.

## 4.2. Overview, Double

| Type          | Time in microseconds | Description                                       |
| ------------- | --------------------:| ------------------------------------------------- |
| CPP_BLOCK     |    2.96              | Plain C++ implementation - baseline               |
| NEON 1 1      |    1.89              | NEON Intrinsics                                   |
| NEON 1 4      |   13.4               | NEON Intrinsics with 4 threads                    |
| VDSP 1 1      |    0.916             | vDSP_DFT_zop_CreateSetupD() & vDSP_DFT_ExecuteD() |

### Remarks
* VDSP runs in sub-micro second. NEON 1 1 version runs about 100% slower than VDSP.
* The overhead of multithreading is too big and not worth it.

# 5. Implementations

This section briefly describes each of the implementations tested with some key points in the code.
Those are executed as part of the test program in [test_fft.cpp](./test_fft.cpp).
The top-level object in the 'main()' function is **TestExecutorFFT512Radix2**, which is a subclass of **TestExecutor found** 
in [../common/test_case_with_time_measurements.h](../common/test_case_with_time_measurements.h).
It manages one single test suite, which consists of test cases.
It arranges the input data, allocates memory, executes each test case multiple times and measures the running times, cleans up, and reports the results.
Each implementation type is implemented as a **TestCaseFFT512Radix2**, which is a subclass of **TestCaseWithTimeMeasurements**
 in [../common/test_case_with_time_measurements.h](../common/test_case_with_time_measurements.h).
The main part is implemented in **TestCaseFFT512Radix2::run()**, and it is the subject for the running time measurements.

The general strategy is as follows:

* Based on [radix-2 Cookey_Tukey recursive algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm).

* Shuffle the input array based on pre-calculated index mapping for the size 512.

* Use table lookup for twiddle factors (trigonometry).

## 5.1. CPP_BLOCK 1 1
[**class TestCaseFFT512Radix2_baseline** in test_fft.cpp](./test_fft.cpp)

The top-level function is as follows:
The first part shuffles the elements of the input *Re* and *Im* arrays.
The main part goes through the butterfly operations layer-by-layer.
```
void cfft_512_forward()
{
    T re1[FFT_LEN_512];
    T im1[FFT_LEN_512];
    T re2[FFT_LEN_512];
    T im2[FFT_LEN_512];

    shuffle_values_512_radix_2( this->m_time_re, re1 );
    shuffle_values_512_radix_2( this->m_time_im, im1 );

    butterfly_one_layer(   1, 256, re1, im1, re2, im2 );
    butterfly_one_layer(   2, 128, re2, im2, re1, im1 );
    butterfly_one_layer(   4,  64, re1, im1, re2, im2 );
    butterfly_one_layer(   8,  32, re2, im2, re1, im1 );
    butterfly_one_layer(  16,  16, re1, im1, re2, im2 );
    butterfly_one_layer(  32,   8, re2, im2, re1, im1 );
    butterfly_one_layer(  64,   4, re1, im1, re2, im2 );
    butterfly_one_layer( 128,   2, re2, im2, re1, im1 );
    butterfly_one_layer( 256,   1, re1, im1, re2, im2 );

    memcpy( this->m_freq_re, re2, sizeof(T) * FFT_LEN_512 );
    memcpy( this->m_freq_im, im2, sizeof(T) * FFT_LEN_512 );
}
```
The butterfly operation per layer is as follows.
```
inline void butterfly_one_layer(

    const int straddle,
    const int trig_table_multiple,

    T* in_re,
    T* in_im,
    T* out_re,
    T* out_im
) {
    for ( int block = 0; block < FFT_LEN_512; block += straddle*2 ) {

        T* in_re_p  = &( in_re[block]  );
        T* in_im_p  = &( in_im[block]  );
        T* out_re_p = &( out_re[block] );
        T* out_im_p = &( out_im[block] );

        for ( int i = 0; i < straddle; i++ ) {
            butterfly_one_pair( straddle, trig_table_multiple, i, in_re_p, in_im_p, out_re_p, out_im_p );
        }
    }
}
```

Finally the main part is as follows.
```
inline void butterfly_one_pair(

    const int straddle,
    const int trig_table_multiple,
    const int index_within_block,

    T* in_re,
    T* in_im,
    T* out_re,
    T* out_im
) {
    const T in_re1    = in_re[ index_within_block            ];
    const T in_im1    = in_im[ index_within_block            ];
    const T in_re2    = in_re[ index_within_block + straddle ];
    const T in_im2    = in_im[ index_within_block + straddle ];

    const int table_index = 0xff & ( trig_table_multiple * index_within_block );

    const T tw_re     = m_cos_0_to_minus_pi_256[ table_index ];
    const T tw_im     = m_sin_0_to_minus_pi_256[ table_index ];

    const T offset_re = tw_re * in_re2 - tw_im * in_im2;
    const T offset_im = tw_re * in_im2 + tw_im * in_re2;

    out_re[ index_within_block            ] = in_re1 + offset_re;
    out_im[ index_within_block            ] = in_im1 + offset_im;
    out_re[ index_within_block + straddle ] = in_re1 - offset_re;
    out_im[ index_within_block + straddle ] = in_im1 - offset_im;
}
```

Please see `class TestCaseFFT512Radix2_baseline` in [test_fft.cpp](./test_fft.cpp) for details.

## 5.2. NEON 1 1 : Neon Intrinsics
[**class TestCaseFFT512Radix2_NEON** in test_ffp.cpp](./test_fft.cpp)

This is based on CPP_BLOCK 1 1 but the pair operations are parallelized using the NEON's lanes.


```
    inline void butterfly_four_pairs_NEON(

        const int straddle,
        const int trig_table_multiple,
        const int index_within_block,

        T* in_re,
        T* in_im,
        T* out_re,
        T* out_im
    ) {
        if constexpr ( is_same< float,T >::value ) {

            const float32x4_t in_re1 = vld1q_f32( &( in_re[ index_within_block            ] ) );
            const float32x4_t in_im1 = vld1q_f32( &( in_im[ index_within_block            ] ) );
            const float32x4_t in_re2 = vld1q_f32( &( in_re[ index_within_block + straddle ] ) );
            const float32x4_t in_im2 = vld1q_f32( &( in_im[ index_within_block + straddle ] ) );

            const int         table_index1 = 0xff & ( trig_table_multiple * index_within_block         );
            const int         table_index2 = 0xff & ( trig_table_multiple * ( index_within_block + 1 ) );
            const int         table_index3 = 0xff & ( trig_table_multiple * ( index_within_block + 2 ) );
            const int         table_index4 = 0xff & ( trig_table_multiple * ( index_within_block + 3 ) );

            const float32x4_t tw_re  = { this->m_cos_0_to_minus_pi_256[ table_index1 ] , 
                                         this->m_cos_0_to_minus_pi_256[ table_index2 ] , 
                                         this->m_cos_0_to_minus_pi_256[ table_index3 ] , 
                                         this->m_cos_0_to_minus_pi_256[ table_index4 ]  };

            const float32x4_t tw_im  = { this->m_sin_0_to_minus_pi_256[ table_index1 ] , 
                                         this->m_sin_0_to_minus_pi_256[ table_index2 ] , 
                                         this->m_sin_0_to_minus_pi_256[ table_index3 ] , 
                                         this->m_sin_0_to_minus_pi_256[ table_index4 ]  };

            // const float offset_re = tw_re * v2_re - tw_im * v2_im;
            const float32x4_t offset_re_part1 = vmulq_f32( tw_re, in_re2 );
            const float32x4_t offset_re       = vmlsq_f32( offset_re_part1, tw_im, in_im2 );

            // const float offset_im = tw_re * v2_im + tw_im * v2_re;
            const float32x4_t offset_im_part1 = vmulq_f32( tw_re, in_im2 );
            const float32x4_t offset_im       = vmlaq_f32( offset_im_part1, tw_im, in_re2 );

            const float32x4_t v1_re = vaddq_f32( in_re1, offset_re );
            const float32x4_t v1_im = vaddq_f32( in_im1, offset_im );
            const float32x4_t v2_re = vsubq_f32( in_re1, offset_re );
            const float32x4_t v2_im = vsubq_f32( in_im1, offset_im );

            vst1q_f32( &( out_re[ index_within_block            ] ), v1_re );
            vst1q_f32( &( out_im[ index_within_block            ] ), v1_im );
            vst1q_f32( &( out_re[ index_within_block + straddle ] ), v2_re );
            vst1q_f32( &( out_im[ index_within_block + straddle ] ), v2_im );
        }
        else {

            const float64x2_t in_re1_1 = vld1q_f64( &( in_re[ index_within_block                ] ) );
            const float64x2_t in_re1_2 = vld1q_f64( &( in_re[ index_within_block + 2            ] ) );
            const float64x2_t in_im1_1 = vld1q_f64( &( in_im[ index_within_block                ] ) );
            const float64x2_t in_im1_2 = vld1q_f64( &( in_im[ index_within_block + 2            ] ) );
            const float64x2_t in_re2_1 = vld1q_f64( &( in_re[ index_within_block + straddle     ] ) );
            const float64x2_t in_re2_2 = vld1q_f64( &( in_re[ index_within_block + straddle + 2 ] ) );
            const float64x2_t in_im2_1 = vld1q_f64( &( in_im[ index_within_block + straddle     ] ) );
            const float64x2_t in_im2_2 = vld1q_f64( &( in_im[ index_within_block + straddle + 2 ] ) );

            const int         table_index1 = 0xff & ( trig_table_multiple * index_within_block         );
            const int         table_index2 = 0xff & ( trig_table_multiple * ( index_within_block + 1 ) );
            const int         table_index3 = 0xff & ( trig_table_multiple * ( index_within_block + 2 ) );
            const int         table_index4 = 0xff & ( trig_table_multiple * ( index_within_block + 3 ) );

            const float64x2_t tw_re_1  = { this->m_cos_0_to_minus_pi_256[ table_index1 ] , 
                                           this->m_cos_0_to_minus_pi_256[ table_index2 ]  }; 
            const float64x2_t tw_re_2  = { this->m_cos_0_to_minus_pi_256[ table_index3 ] , 
                                           this->m_cos_0_to_minus_pi_256[ table_index4 ]  };

            const float64x2_t tw_im_1  = { this->m_sin_0_to_minus_pi_256[ table_index1 ] , 
                                           this->m_sin_0_to_minus_pi_256[ table_index2 ] };
            const float64x2_t tw_im_2  = { this->m_sin_0_to_minus_pi_256[ table_index3 ] , 
                                           this->m_sin_0_to_minus_pi_256[ table_index4 ]  };

            // const float offset_re = tw_re * v2_re - tw_im * v2_im;
            const float64x2_t offset_re_part1_1 = vmulq_f64( tw_re_1, in_re2_1 );
            const float64x2_t offset_re_part1_2 = vmulq_f64( tw_re_2, in_re2_2 );
            const float64x2_t offset_re_1       = vmlsq_f64( offset_re_part1_1, tw_im_1, in_im2_1 );
            const float64x2_t offset_re_2       = vmlsq_f64( offset_re_part1_2, tw_im_2, in_im2_2 );

            // const float offset_im = tw_re * v2_im + tw_im * v2_re;
            const float64x2_t offset_im_part1_1 = vmulq_f64( tw_re_1, in_im2_1 );
            const float64x2_t offset_im_part1_2 = vmulq_f64( tw_re_2, in_im2_2 );
            const float64x2_t offset_im_1       = vmlaq_f64( offset_im_part1_1, tw_im_1, in_re2_1 );
            const float64x2_t offset_im_2       = vmlaq_f64( offset_im_part1_2, tw_im_2, in_re2_2 );

            const float64x2_t v1_re_1 = vaddq_f64( in_re1_1, offset_re_1 );
            const float64x2_t v1_re_2 = vaddq_f64( in_re1_2, offset_re_2 );
            const float64x2_t v1_im_1 = vaddq_f64( in_im1_1, offset_im_1 );
            const float64x2_t v1_im_2 = vaddq_f64( in_im1_2, offset_im_2 );
            const float64x2_t v2_re_1 = vsubq_f64( in_re1_1, offset_re_1 );
            const float64x2_t v2_re_2 = vsubq_f64( in_re1_2, offset_re_2 );
            const float64x2_t v2_im_1 = vsubq_f64( in_im1_1, offset_im_1 );
            const float64x2_t v2_im_2 = vsubq_f64( in_im1_2, offset_im_2 );

            vst1q_f64( &( out_re[ index_within_block                ] ), v1_re_1 );
            vst1q_f64( &( out_re[ index_within_block + 2            ] ), v1_re_2 );
            vst1q_f64( &( out_im[ index_within_block                ] ), v1_im_1 );
            vst1q_f64( &( out_im[ index_within_block + 2            ] ), v1_im_2 );
            vst1q_f64( &( out_re[ index_within_block + straddle     ] ), v2_re_1 );
            vst1q_f64( &( out_re[ index_within_block + straddle + 2 ] ), v2_re_2 );
            vst1q_f64( &( out_im[ index_within_block + straddle     ] ), v2_im_1 );
            vst1q_f64( &( out_im[ index_within_block + straddle + 2 ] ), v2_im_2 );
        }
    }
```

Please see `class TestCaseFFT512Radix2_NEON` in [test_fft.cpp](./test_fft.cpp) for details.

## 5.3. NEON 1 4: NEON Intrinsics with 4 threads.
[**class TestCaseFFT512Radix2_multithread_NEON** in test_fft.cpp](./test_fft.cpp)

This is based on NEON 1 1, but the per-layer operations up to the straddle of 64 are parallelized over 4 threads.
The ramaining 2 layers are handles in the main thread.

Following is the thread definition to process the layers of straddle up to 64.
```
auto thread_lambda = [this]( const size_t thread_index ) {

    while ( true ) {

        m_fan_out.wait( thread_index );
        if( m_fan_out.isTerminating() ) {
            break;
        }

        butterfly_one_layer_multithread4     (   1, 256, re1, im1, re2, im2, thread_index );
        butterfly_one_layer_multithread4     (   2, 128, re2, im2, re1, im1, thread_index );
        butterfly_one_layer_multithread4_NEON(   4,  64, re1, im1, re2, im2, thread_index );
        butterfly_one_layer_multithread4_NEON(   8,  32, re2, im2, re1, im1, thread_index );
        butterfly_one_layer_multithread4_NEON(  16,  16, re1, im1, re2, im2, thread_index );
        butterfly_one_layer_multithread4_NEON(  32,   8, re2, im2, re1, im1, thread_index );
        butterfly_one_layer_multithread4_NEON(  64,   4, re1, im1, re2, im2, thread_index );

        m_fan_in.notify();
        if( m_fan_in.isTerminating() ) {
            break;
        }
    }  
};
```

And the main thread waits for the 4 threads to finish, and takes care of the ramaining 2 layers as follows.

```
virtual void run() {

    this->shuffle_values_512_radix_2( this->m_time_re, re1 );
    this->shuffle_values_512_radix_2( this->m_time_im, im1 );

    m_fan_out.notify();
    m_fan_in. wait();

    this->butterfly_one_layer_NEON( 128,   2, re2, im2, re1, im1 );
    this->butterfly_one_layer_NEON( 256,   1, re1, im1, re2, im2 );

    memcpy( this->m_freq_re, re2, sizeof(T) * FFT_LEN_512 );
    memcpy( this->m_freq_im, im2, sizeof(T) * FFT_LEN_512 );
}
```

Please see `class TestCaseFFT512Radix2_multithread_NEON` in [test_fft.cpp](./test_fft.cpp) for details.

## 5.4. VDSP 1 1
[**class TestCaseFFT512Radix2_vDSP** in test_fft.cpp](./test_fft.cpp)

It simply calls the following functions.

Float:

```
vDSP_DFT_Setup  setup = vDSP_DFT_zop_CreateSetup( nullptr, FFT_LEN_512, vDSP_DFT_FORWARD );

vDSP_DFT_Execute( setup, this->m_time_re, this->m_time_im, this->m_freq_re, this->m_freq_im );
```

Double:
```
vDSP_DFT_SetupD  setup = vDSP_DFT_zop_CreateSetup( nullptr, FFT_LEN_512, vDSP_DFT_FORWARD );

vDSP_DFT_ExecuteD( setup, this->m_time_re, this->m_time_im, this->m_freq_re, this->m_freq_im );
```

Please see `class TestCaseFFT512Radix2_vDSP` in [test_fft.cpp](./test_fft.cpp) for details.

## 5.5. METAL 0 0
[**class TestCaseFFT512Radix2_metal** in test_fft.cpp](./test_fft.cpp)

It uses *threadgroup memory* for the shuffled *Re* and *Im* arrays as well as pre-calculated cos, sin tables.
Also it utilizes `simd_shuffle_xor()` for the butterfly operations.

Please see ` fft_512_radix_2()` in [metal/fft.metal](./metal/fft.metal) for details.


# 6. References

* Digital signal processing" by Proakis, Manolakis 4th edition Chap 8: Efficient Computation of the DFT: Fast Fourier Transform

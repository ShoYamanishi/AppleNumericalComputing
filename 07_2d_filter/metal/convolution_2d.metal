#include <metal_stdlib>

using namespace metal;

struct convolution_2d_constants {
    uint num_elements;
    uint num_elements_stage_2;
    uint image_width;
    uint image_height;
    uint kernel_size;
};

static inline int index_from_xy( const thread int x, const thread int y, const int width ){
    return y * width + x;
}

kernel void convolution_2d_naive (

    device float*       input_image                        [[ buffer(0) ]],

    device const float* conv_kernel                        [[ buffer(1) ]],

    device float*       output_image                       [[ buffer(2) ]],

    device const struct convolution_2d_constants&
                        constants                          [[ buffer(3) ]],

    const        uint   thread_position_in_grid            [[ thread_position_in_grid ]]

)
{
    if (thread_position_in_grid < constants.num_elements ) {

        const uint KERN_OFFSET = constants.kernel_size/2;

        const uint center_x = thread_position_in_grid % constants.image_width;

        const uint center_y = thread_position_in_grid / constants.image_width;

        float sum = 0.0;

        for ( int kern_y = 0; kern_y < (int)constants.kernel_size; kern_y++ ) {

            const int image_y = ( kern_y - KERN_OFFSET ) + center_y;

            if ( 0 <= image_y && image_y < (int)constants.image_height ) {

                for ( int kern_x = 0; kern_x < (int)constants.kernel_size; kern_x++ ) {

                    const int image_x = ( kern_x - KERN_OFFSET ) + center_x;

                    const float v =   ( 0 <= image_x && image_x < (int)constants.image_width )

                                    ? (   conv_kernel[ index_from_xy(kern_x,  kern_y,  constants.kernel_size) ]
                                        * input_image[ index_from_xy(image_x, image_y, constants.image_width) ]
                                      )
                                    : 0.0
                                    ;
                    sum += v;
                }
            }
        }

        output_image[ index_from_xy(center_x, center_y, constants.image_width) ] = sum;
    }
}


kernel void convolution_5x5_stage1 (

    device float*       input_image                        [[ buffer(0) ]],

    device const float* conv_kernel                        [[ buffer(1) ]],

    device float*       output_image                       [[ buffer(2) ]],

    device const struct convolution_2d_constants&
                        constants                          [[ buffer(3) ]],

    const        uint   thread_position_in_grid            [[ thread_position_in_grid ]],

    const        uint   thread_position_in_threadgroup     [[ thread_position_in_threadgroup ]]

)
{
    threadgroup float k[5][5];// row major k[y][x]

    if ( thread_position_in_grid < constants.num_elements ) {

        const int width  = constants.image_width;
        const int height = constants.image_height;

        // copying the kernel to the threadgroup(scratch) memory 

        const uint kern_x  = thread_position_in_threadgroup % constants.image_width;
        const uint kern_y  = thread_position_in_threadgroup / constants.image_width;
        if ( thread_position_in_threadgroup < 25 ) 
        {
            k[kern_y][kern_x] = conv_kernel[thread_position_in_threadgroup];
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // in the following the variables are in v[y][x]. e. v13 means row 1, col 3.

        const int pos_y  = thread_position_in_grid / constants.image_width;

        if ( pos_y == 0 ) {

            float sum = 0.0;
            // one thread accesses coalesced memory 3 times.
            const float v33 = input_image[ thread_position_in_grid             ];
            const float v43 = input_image[ thread_position_in_grid +     width ];
            const float v53 = input_image[ thread_position_in_grid + 2 * width ];

            sum += ( v33 * k[2][2] + v43 * k[3][2] + v53 * k[4][2] );

            const float v31 = simd_shuffle_up( v33, 2 );
            const float v41 = simd_shuffle_up( v43, 2 );
            const float v51 = simd_shuffle_up( v53, 2 );

            sum += ( v31 * k[2][0] + v41 * k[3][0] + v51 * k[4][0] );

            const float v32 = simd_shuffle_up( v33, 1 );
            const float v42 = simd_shuffle_up( v43, 1 );
            const float v52 = simd_shuffle_up( v53, 1 );

            sum += ( v32 * k[2][1] + v42 * k[3][1] + v52 * k[4][1] );

            const float v34 = simd_shuffle_down( v33, 1 );
            const float v44 = simd_shuffle_down( v43, 1 );
            const float v54 = simd_shuffle_down( v53, 1 );

            sum += ( v34 * k[2][3] + v44 * k[3][3] + v54 * k[4][3] );

            const float v35 = simd_shuffle_down( v33, 2 );
            const float v45 = simd_shuffle_down( v43, 2 );
            const float v55 = simd_shuffle_down( v53, 2 );

            sum += ( v35 * k[2][4] + v45 * k[3][4] + v55 * k[4][4] );

            output_image[ thread_position_in_grid ] = sum;
        
        }
        else if ( pos_y == 1 ) {

            float sum = 0.0;

            // one thread accesses coalesced memory 4 times.
            const float v23 = input_image[ thread_position_in_grid -     width ];
            const float v33 = input_image[ thread_position_in_grid             ];
            const float v43 = input_image[ thread_position_in_grid +     width ];
            const float v53 = input_image[ thread_position_in_grid + 2 * width ];

            sum += ( v23 * k[1][2] + v33 * k[2][2] + v43 * k[3][2] + v53 * k[4][2] );

            const float v21 = simd_shuffle_up( v23, 2 );
            const float v31 = simd_shuffle_up( v33, 2 );
            const float v41 = simd_shuffle_up( v43, 2 );
            const float v51 = simd_shuffle_up( v53, 2 );

            sum += ( v21 * k[1][0] + v31 * k[2][0] + v41 * k[3][0] + v51 * k[4][0] );

            const float v22 = simd_shuffle_up( v23, 1 );
            const float v32 = simd_shuffle_up( v33, 1 );
            const float v42 = simd_shuffle_up( v43, 1 );
            const float v52 = simd_shuffle_up( v53, 1 );

            sum += ( v22 * k[1][1] + v32 * k[2][1] + v42 * k[3][1] + v52 * k[4][1] );

            const float v24 = simd_shuffle_down( v23, 1 );
            const float v34 = simd_shuffle_down( v33, 1 );
            const float v44 = simd_shuffle_down( v43, 1 );
            const float v54 = simd_shuffle_down( v53, 1 );

            sum += ( v24 * k[1][3] + v34 * k[2][3] + v44 * k[3][3] + v54 * k[4][3] );

            const float v25 = simd_shuffle_down( v23, 2 );
            const float v35 = simd_shuffle_down( v33, 2 );
            const float v45 = simd_shuffle_down( v43, 2 );
            const float v55 = simd_shuffle_down( v53, 2 );

            sum += ( v25 * k[1][4] + v35 * k[2][4] + v45 * k[3][4] + v55 * k[4][4] );

            output_image[ thread_position_in_grid ] = sum;

        }
        else if ( pos_y == height - 2 ) {

            float sum = 0.0;

            // one thread accesses coalesced memory 5 times.
            const float v13 = input_image[ thread_position_in_grid - 2 * width ];
            const float v23 = input_image[ thread_position_in_grid -     width ];
            const float v33 = input_image[ thread_position_in_grid             ];
            const float v43 = input_image[ thread_position_in_grid +     width ];

            sum += ( v13 * k[0][2] + v23 * k[1][2] + v33 * k[2][2] + v43 * k[3][2] );

            const float v11 = simd_shuffle_up( v13, 2 );
            const float v21 = simd_shuffle_up( v23, 2 );
            const float v31 = simd_shuffle_up( v33, 2 );
            const float v41 = simd_shuffle_up( v43, 2 );

            sum += ( v11 * k[0][0] + v21 * k[1][0] + v31 * k[2][0] + v41 * k[3][0] );

            const float v12 = simd_shuffle_up( v13, 1 );
            const float v22 = simd_shuffle_up( v23, 1 );
            const float v32 = simd_shuffle_up( v33, 1 );
            const float v42 = simd_shuffle_up( v43, 1 );

            sum += ( v12 * k[0][1] + v22 * k[1][1] + v32 * k[2][1] + v42 * k[3][1] );

            const float v14 = simd_shuffle_down( v13, 1 );
            const float v24 = simd_shuffle_down( v23, 1 );
            const float v34 = simd_shuffle_down( v33, 1 );
            const float v44 = simd_shuffle_down( v43, 1 );

            sum += ( v14 * k[0][3] + v24 * k[1][3] + v34 * k[2][3] + v44 * k[3][3] );

            const float v15 = simd_shuffle_down( v13, 2 );
            const float v25 = simd_shuffle_down( v23, 2 );
            const float v35 = simd_shuffle_down( v33, 2 );
            const float v45 = simd_shuffle_down( v43, 2 );

            sum += ( v15 * k[0][4] + v25 * k[1][4] + v35 * k[2][4] + v45 * k[3][4] );

            output_image[ thread_position_in_grid ] = sum;

        }
        else if ( pos_y == height - 1 ) {

            float sum = 0.0;

            // one thread accesses coalesced memory 5 times.
            const float v13 = input_image[ thread_position_in_grid - 2 * width ];
            const float v23 = input_image[ thread_position_in_grid -     width ];
            const float v33 = input_image[ thread_position_in_grid             ];

            sum += ( v13 * k[0][2] + v23 * k[1][2] + v33 * k[2][2] );

            const float v11 = simd_shuffle_up( v13, 2 );
            const float v21 = simd_shuffle_up( v23, 2 );
            const float v31 = simd_shuffle_up( v33, 2 );

            sum += ( v11 * k[0][0] + v21 * k[1][0] + v31 * k[2][0] );

            const float v12 = simd_shuffle_up( v13, 1 );
            const float v22 = simd_shuffle_up( v23, 1 );
            const float v32 = simd_shuffle_up( v33, 1 );

            sum += ( v12 * k[0][1] + v22 * k[1][1] + v32 * k[2][1] );

            const float v14 = simd_shuffle_down( v13, 1 );
            const float v24 = simd_shuffle_down( v23, 1 );
            const float v34 = simd_shuffle_down( v33, 1 );

            sum += ( v14 * k[0][3] + v24 * k[1][3] + v34 * k[2][3] );

            const float v15 = simd_shuffle_down( v13, 2 );
            const float v25 = simd_shuffle_down( v23, 2 );
            const float v35 = simd_shuffle_down( v33, 2 );

            sum += ( v15 * k[0][4] + v25 * k[1][4] + v35 * k[2][4] );

            output_image[ thread_position_in_grid ] = sum;

        }
        else {
            float sum = 0.0;

            // one thread accesses coalesced memory 5 times.
            const float v13 = input_image[ thread_position_in_grid - 2 * width ];
            const float v23 = input_image[ thread_position_in_grid -     width ];
            const float v33 = input_image[ thread_position_in_grid             ];
            const float v43 = input_image[ thread_position_in_grid +     width ];
            const float v53 = input_image[ thread_position_in_grid + 2 * width ];

            sum += ( v13 * k[0][2] + v23 * k[1][2] + v33 * k[2][2] + v43 * k[3][2] + v53 * k[4][2] );

            const float v11 = simd_shuffle_up( v13, 2 );
            const float v21 = simd_shuffle_up( v23, 2 );
            const float v31 = simd_shuffle_up( v33, 2 );
            const float v41 = simd_shuffle_up( v43, 2 );
            const float v51 = simd_shuffle_up( v53, 2 );

            sum += ( v11 * k[0][0] + v21 * k[1][0] + v31 * k[2][0] + v41 * k[3][0] + v51 * k[4][0] );

            const float v12 = simd_shuffle_up( v13, 1 );
            const float v22 = simd_shuffle_up( v23, 1 );
            const float v32 = simd_shuffle_up( v33, 1 );
            const float v42 = simd_shuffle_up( v43, 1 );
            const float v52 = simd_shuffle_up( v53, 1 );

            sum += ( v12 * k[0][1] + v22 * k[1][1] + v32 * k[2][1] + v42 * k[3][1] + v52 * k[4][1] );

            const float v14 = simd_shuffle_down( v13, 1 );
            const float v24 = simd_shuffle_down( v23, 1 );
            const float v34 = simd_shuffle_down( v33, 1 );
            const float v44 = simd_shuffle_down( v43, 1 );
            const float v54 = simd_shuffle_down( v53, 1 );

            sum += ( v14 * k[0][3] + v24 * k[1][3] + v34 * k[2][3] + v44 * k[3][3] + v54 * k[4][3] );

            const float v15 = simd_shuffle_down( v13, 2 );
            const float v25 = simd_shuffle_down( v23, 2 );
            const float v35 = simd_shuffle_down( v33, 2 );
            const float v45 = simd_shuffle_down( v43, 2 );
            const float v55 = simd_shuffle_down( v53, 2 );

            sum += ( v15 * k[0][4] + v25 * k[1][4] + v35 * k[2][4] + v45 * k[3][4] + v55 * k[4][4] );

            output_image[ thread_position_in_grid ] = sum;
        }
    }
}

// NOTE!!! The following kernel does not work with threadgroup of 1024 threads
//         It works with 768 threads on Apple M1. This seems to be due to too many
//         local variables per thread.
kernel void convolution_5x5_stage2 (

    device float*       input_image                        [[ buffer(0) ]],

    device const float* conv_kernel                        [[ buffer(1) ]],

    device float*       output_image                       [[ buffer(2) ]],

    device const struct convolution_2d_constants&
                        constants                          [[ buffer(3) ]],

    const        uint   thread_position_in_grid            [[ thread_position_in_grid ]],

    const        uint   thread_position_in_threadgroup     [[ thread_position_in_threadgroup ]]
)
{
    threadgroup float k[5][5];// row major k[y][x]

    if ( thread_position_in_grid < constants.num_elements_stage_2 ) {


        const int width  = constants.image_width;
        const int height = constants.image_height;

        // copying the kernel to the threadgroup(scratch) memory 
        const uint kern_x  = thread_position_in_threadgroup % constants.image_width;
        const uint kern_y  = thread_position_in_threadgroup / constants.image_width;

        if ( thread_position_in_threadgroup < 25 ) 
        {
            k[kern_y][kern_x] = conv_kernel[thread_position_in_threadgroup];
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        // thread_position_in_grid: 0   1   2   3   4   5   6   7   8   9  ...
        // adjusted_linear_pos:     0   1  30  31  32  33  60  61  62  63  ...

        const int adjusted_linear_pos =    ((thread_position_in_grid + 2) / 4) * 32 
                                         + ((thread_position_in_grid + 2) % 4) 
                                         - 2;

        const int pos_x = adjusted_linear_pos % width;
        const int pos_y = adjusted_linear_pos / width;

        const bool x_minus_2_ok = (pos_x >= 2);
        const bool x_minus_1_ok = (pos_x >= 1);
        const bool x_plus_1_ok  = (pos_x < width - 1);
        const bool x_plus_2_ok  = (pos_x < width - 2);

        const bool y_minus_2_ok = (pos_y >= 2);
        const bool y_minus_1_ok = (pos_y >= 1);
        const bool y_plus_1_ok  = (pos_y < height - 1);
        const bool y_plus_2_ok  = (pos_y < height - 2);

        const float v11 = (x_minus_2_ok && y_minus_2_ok)? input_image[(pos_y - 2) * width + pos_x - 2 ]: 0.0;
        const float v12 = (x_minus_1_ok && y_minus_2_ok)? input_image[(pos_y - 2) * width + pos_x - 1 ]: 0.0;
        const float v13 = (                y_minus_2_ok)? input_image[(pos_y - 2) * width + pos_x     ]: 0.0;
        const float v14 = (x_plus_1_ok  && y_minus_2_ok)? input_image[(pos_y - 2) * width + pos_x + 1 ]: 0.0;
        const float v15 = (x_plus_2_ok  && y_minus_2_ok)? input_image[(pos_y - 2) * width + pos_x + 2 ]: 0.0;

        const float v21 = (x_minus_2_ok && y_minus_1_ok)? input_image[(pos_y - 1) * width + pos_x - 2 ]: 0.0;
        const float v22 = (x_minus_1_ok && y_minus_1_ok)? input_image[(pos_y - 1) * width + pos_x - 1 ]: 0.0;
        const float v23 = (                y_minus_1_ok)? input_image[(pos_y - 1) * width + pos_x     ]: 0.0;
        const float v24 = (x_plus_1_ok  && y_minus_1_ok)? input_image[(pos_y - 1) * width + pos_x + 1 ]: 0.0;
        const float v25 = (x_plus_2_ok  && y_minus_1_ok)? input_image[(pos_y - 1) * width + pos_x + 2 ]: 0.0;

        const float v31 = (x_minus_2_ok                )? input_image[ pos_y      * width + pos_x - 2 ]: 0.0;
        const float v32 = (x_minus_1_ok                )? input_image[ pos_y      * width + pos_x - 1 ]: 0.0;
        const float v33 =                                 input_image[ pos_y      * width + pos_x     ];
        const float v34 = (x_plus_1_ok                 )? input_image[ pos_y      * width + pos_x + 1 ]: 0.0;
        const float v35 = (x_plus_2_ok                 )? input_image[ pos_y      * width + pos_x + 2 ]: 0.0;

        const float v41 = (x_minus_2_ok && y_plus_1_ok )? input_image[(pos_y + 1) * width + pos_x - 2 ]: 0.0;
        const float v42 = (x_minus_1_ok && y_plus_1_ok )? input_image[(pos_y + 1) * width + pos_x - 1 ]: 0.0;
        const float v43 = (                y_plus_1_ok )? input_image[(pos_y + 1) * width + pos_x     ]: 0.0;
        const float v44 = (x_plus_1_ok  && y_plus_1_ok )? input_image[(pos_y + 1) * width + pos_x + 1 ]: 0.0;
        const float v45 = (x_plus_2_ok  && y_plus_1_ok )? input_image[(pos_y + 1) * width + pos_x + 2 ]: 0.0;

        const float v51 = (x_minus_2_ok && y_plus_2_ok )? input_image[(pos_y + 2) * width + pos_x - 2 ]: 0.0;
        const float v52 = (x_minus_1_ok && y_plus_2_ok )? input_image[(pos_y + 2) * width + pos_x - 1 ]: 0.0;
        const float v53 = (                y_plus_2_ok )? input_image[(pos_y + 2) * width + pos_x     ]: 0.0;
        const float v54 = (x_plus_1_ok  && y_plus_2_ok )? input_image[(pos_y + 2) * width + pos_x + 1 ]: 0.0;
        const float v55 = (x_plus_2_ok  && y_plus_2_ok )? input_image[(pos_y + 2) * width + pos_x + 2 ]: 0.0;

        output_image[ pos_y * width + pos_x ] 
            =    v11 * k[0][0] + v12 * k[0][1] + v13 * k[0][2] + v14 * k[0][3] + v15 * k[0][4]
              +  v21 * k[1][0] + v22 * k[1][1] + v23 * k[1][2] + v24 * k[1][3] + v25 * k[1][4] 
              +  v31 * k[2][0] + v32 * k[2][1] + v33 * k[2][2] + v34 * k[2][3] + v35 * k[2][4] 
              +  v41 * k[3][0] + v42 * k[3][1] + v43 * k[3][2] + v44 * k[3][3] + v45 * k[3][4] 
              +  v51 * k[4][0] + v52 * k[4][1] + v53 * k[4][2] + v54 * k[4][3] + v55 * k[4][4] ;

    }

}

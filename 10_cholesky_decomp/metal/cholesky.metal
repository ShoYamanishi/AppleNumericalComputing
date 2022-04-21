#include <metal_stdlib>

using namespace metal;

struct cholesky_constants {
    uint  dim;
};



static inline int lower_mat_index( const int i, const int j, const int dim )
{
    const int num_elems = (dim + 1) * dim / 2;
    const int i_rev = (dim -1) - i;
    const int j_rev = (dim -1) - j;
    return num_elems - 1 - ( j_rev * (j_rev + 1) / 2 + i_rev );
}

// Column-Cholesky
// - outer-loop is over column
// - at each column each thread takes dim/1024 rows.
// - innter-loop is over the left columns.
kernel void decompose_cholesky (

    device float*                    L                              [[ buffer(0) ]],

    device const cholesky_constants& constants                      [[ buffer(1) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]
) {

    threadgroup float recip_a_jj;

    threadgroup float col_cache[1024];

    const int DIM = (int)constants.dim;

    for ( int col = 0; col < DIM; col++ ) {

        if ( col > 0 ) {
            threadgroup_barrier(  mem_flags::mem_device );
        }

        const int first_valid_row_pos = ((int)(col / 1024)) * 1024;

        const int remaining_col = DIM - col;

        for ( int iter_row_pos = first_valid_row_pos; iter_row_pos < DIM; iter_row_pos += 1024 ) {

            const int row = iter_row_pos + thread_position_in_threadgroup;

            if ( row >= col && row < DIM ) {

                const int Lpos = lower_mat_index( row, col, DIM );
                
                float Lval = L[ Lpos ];
                
                for ( int k = 0; k < col; k++ ) {

                    Lval -= ( L [ lower_mat_index( row, k, DIM ) ] * L [ lower_mat_index( col, k, DIM ) ] );
                }

                if ( col == row ) {

                    Lval = sqrt(Lval);
                    recip_a_jj = 1.0 / Lval;
                }

                if ( remaining_col <= 1024 ) {
                    col_cache[ thread_position_in_threadgroup ] = Lval;
                }
                else {
                    L [ Lpos ] = Lval;
                }

            }
        }

        if ( remaining_col > 1024 ) {

            threadgroup_barrier(  mem_flags::mem_device );
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        for ( int iter_row_pos = first_valid_row_pos; iter_row_pos < DIM; iter_row_pos += 1024 ) {

            const int row = iter_row_pos + thread_position_in_threadgroup;

            const int Lpos = lower_mat_index( row, col, DIM );

            if ( remaining_col > 1024 ) {

                if ( row > col && row < DIM ) {

                    L [ Lpos ] = L [ Lpos ] * recip_a_jj;
                }
            }
            else {

                if ( row >= col && row < DIM ) {
                    if (row == col) {
                        L [ Lpos ] = col_cache[ thread_position_in_threadgroup ];
                    }
                    else {
                        L [ Lpos ] = col_cache[ thread_position_in_threadgroup ] * recip_a_jj;
                    }
                }
            }
        }
    }
}


kernel void solve_Lyeb (

    device float*                    L                              [[ buffer(0) ]],

    device float*                    y                              [[ buffer(1) ]],

    device float*                    b                              [[ buffer(2) ]],

    device const cholesky_constants& constants                      [[ buffer(3) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]
) {
    threadgroup float sum_cache[1024];

    const int DIM = (int)constants.dim;

    for ( int row = 0; row < DIM; row++ ) {

        // local sum
        float sum = 0.0;
        for ( int col_pos_offset = 0; col_pos_offset < row; col_pos_offset += 1024 ) {

            const int col_pos = col_pos_offset + thread_position_in_threadgroup;
            if (col_pos < row ) {
                sum += ( L [ lower_mat_index( row, col_pos, DIM ) ] * y[col_pos] );
            }
        }

        sum_cache[ thread_position_in_threadgroup ] = sum;
        threadgroup_barrier( mem_flags::mem_threadgroup );

        // reduce
        for ( int i = 512; i > 0 ; i = i >> 1 ) {
            if ( (int)thread_position_in_threadgroup < i ) {

                sum_cache[ thread_position_in_threadgroup ] += sum_cache[ thread_position_in_threadgroup + i ];
            }
            threadgroup_barrier( mem_flags::mem_threadgroup );
        }

        if ( thread_position_in_threadgroup == 0 ) {
            y[row] = ( b[row] - sum_cache[ thread_position_in_threadgroup ] )/ L [ lower_mat_index( row, row, DIM ) ];
        }
    }
}


kernel void solve_Ltxey (

    device float*                    L                              [[ buffer(0) ]],

    device float*                    x                              [[ buffer(1) ]],

    device float*                    y                              [[ buffer(2) ]],

    device const cholesky_constants& constants                      [[ buffer(3) ]],

    const        uint                thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],

    const        uint                threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],

    const        uint                thread_position_in_grid        [[ thread_position_in_grid ]]
) {

    threadgroup float sum_cache[1024];

    const int DIM = (int)constants.dim;

    for ( int row = 0; row < DIM; row++ ) {

        // local sum
        float sum = 0.0;
        for ( int col_pos_offset = 0; col_pos_offset < row; col_pos_offset += 1024 ) {

            const int col_pos = col_pos_offset + thread_position_in_threadgroup;

            if ( col_pos < row ) {

                sum += (   L [ lower_mat_index( ( DIM - 1 ) - col_pos, ( DIM - 1 ) - row, DIM ) ]
                         * x [ (DIM - 1) - col_pos ]
                       );
            }
        }

        sum_cache[ thread_position_in_threadgroup ] = sum;
        threadgroup_barrier( mem_flags::mem_threadgroup );

        // reduce
        for ( int i = 512; i > 0 ; i = i >> 1 ) {
            if ( (int)thread_position_in_threadgroup < i ) {

                sum_cache[ thread_position_in_threadgroup ] += sum_cache[ thread_position_in_threadgroup + i ];
            }
            threadgroup_barrier( mem_flags::mem_threadgroup );
        }

        if ( thread_position_in_threadgroup == 0 ) {
            x[ (DIM - 1) - row ] =   ( y[ (DIM -1) - row ] - sum_cache[ thread_position_in_threadgroup ] )
                                   / L [ lower_mat_index( (DIM - 1) - row, (DIM - 1) - row, DIM ) ];
        }
    }
}

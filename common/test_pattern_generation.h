#ifndef __TEST_PATTERN_GENERATION_H__
#define __TEST_PATTERN_GENERATION_H__

#include "test_case_with_time_measurements.h"

#include <type_traits>

using namespace std;

template<class T>
static void fillArrayWithRandomValues(
    default_random_engine& e,
    T* const               a,
    const size_t           s,
    const T                low,
    const T                high
) {
    static_type_guard<T>();

    if constexpr ( is_same<float, T>::value || is_same<double, T>::value ) {
        uniform_real_distribution dist{ low, high };
        for ( size_t i = 0; i < s; i++ ) {
            a[i] = dist(e);
        }
    }                  
    else {
        uniform_int_distribution dist{ low, high };
        for ( size_t i = 0; i < s; i++ ) {
            a[i] = dist(e);
        }
    }
}

template<class T>
static T getRandomNum(
    default_random_engine& e,
    const T                low,
    const T                high
) {
    static_type_guard<T>();

    if constexpr ( is_same<float, T>::value || is_same<double, T>::value ) {
        uniform_real_distribution dist{ low, high };
        return dist(e);
    }                  
    else {
        uniform_int_distribution dist{ low, high };
        return dist(e);
    }
}


// generate a random sparse matrix in CSR (compressed sparse row) format.
template<class T>
static void generateCSR(

    const int   M,
    const int   N,
    const int   num_nonzero_elems,
    const T     val_min,
    const T     val_max,
    default_random_engine& e,

    int* row_ptrs,
    int* columns,
    T*   values,
    T*   lhs_vector
) {
    uniform_int_distribution  dist_M  { 0,       M - 1   };
    uniform_int_distribution  dist_N  { 0,       N - 1   };
    uniform_real_distribution dist_val{ val_min, val_max };
    vector< std::map<int, T> > nz_cols_per_row;

    for ( int i = 0; i < M; i++ ) {
        nz_cols_per_row.emplace_back();
    }
    for ( int i = 0 ; i < num_nonzero_elems; ) {

        int m   = dist_M( e );
        int n   = dist_N( e );
        T   val = dist_val( e );

        auto& row = nz_cols_per_row[m];
        if ( row.find(n) == row.end() ) {

            row.emplace( make_pair(n, val) );
            i++;
        }
    }

    int sum = 0;
    row_ptrs[0] = 0;
    for ( int i = 0; i < M; i++ ) {

        sum += nz_cols_per_row[i].size();
        row_ptrs[i+1] = sum;
    }

    int pos = 0;
    for ( int i = 0; i < M ; i++ ) {

        auto& row = nz_cols_per_row[i];

        for ( auto it = row.begin(); it != row.end(); it++ ) {

            columns[pos] = it->first;
            values [pos] = it->second;
            pos++;
        }
    }

    for ( int i = 0; i < N; i++ ) {
        lhs_vector[i] = dist_val( e );
    }
}


template<class T>
static void generateDenseMatrixVector (
    const int   M,
    const int   N,
    T*          mat,
    T*          vec,
    const T     val_min,
    const T     val_max,
    std::default_random_engine& e
) {
    std::uniform_real_distribution dist{ val_min, val_max };

    for ( int i = 0; i < M*N; i++ ) {
        mat[i] =  dist( e );
    }

    for ( int i = 0; i < N; i++ ) {
        vec[i] = dist( e );
    }
}


template<class T, bool IS_COL_MAJOR>
static void generateRandomPDLowerMat( T* A, const int dim, const T cond_num, std::default_random_engine& e )
{
    std::uniform_real_distribution<T> dist_positive {  1.0001, cond_num };
    std::uniform_real_distribution<T> dist_any      { -0.9999, 0.9999   };

    T* D = new T[ dim ];
    T* Q = new T[ (dim+1) * dim / 2 ];

    for ( int i = 0; i < dim; i++ ) {

        D[i] = dist_positive( e );
    }

    for ( int i = 0; i < dim; i++ ) {

        for ( int j = 0; j <= i; j++ ) {

            const T val = dist_any( e );

            Q  [ lower_mat_index<IS_COL_MAJOR>( i, j, dim ) ] = val;
        }
    }

    for ( int i = 0 ; i < dim ; i++ ) {

        for ( int j = 0; j <= i; j++ ) {

            T sum = 0.0;

            for ( int k = 0; k < dim; k++ ) {

                const T Q_ik  = (i>=k) ? Q[ lower_mat_index<IS_COL_MAJOR>( i, k, dim ) ] : Q[ lower_mat_index<IS_COL_MAJOR>( k, i, dim ) ];
                const T Qt_kj = (j>=k) ? Q[ lower_mat_index<IS_COL_MAJOR>( j, k, dim ) ] : Q[ lower_mat_index<IS_COL_MAJOR>( k, j, dim ) ];

                sum += ( Q_ik * Qt_kj );
            }

            A [ lower_mat_index<IS_COL_MAJOR>( i, j, dim ) ] = sum;
        }
    }

    // check and correct to strict diagonal dominance.
    // add diagonal elements.
    for ( int i = 0 ; i < dim ; i++ ) {

        T sum = 0.0;
        for ( int j = 0; j < i; j++ ) {
            sum += abs(( A[ lower_mat_index<IS_COL_MAJOR>( i, j, dim ) ] ));
        }
        for ( int j = i+1; j < dim; j++ ) {
            sum += abs(( A[ lower_mat_index<IS_COL_MAJOR>( j, i, dim ) ] ));
        }
        T diff = A[ lower_mat_index<IS_COL_MAJOR>( i, i, dim ) ] - sum;

        if ( diff < 0.0 ) {

            // "A[i] is too small. Correcting.

            A[ lower_mat_index<IS_COL_MAJOR>( i, i, dim ) ] -= diff;
        }
        A[ lower_mat_index<IS_COL_MAJOR>( i, i, dim ) ] *= D[i];

    }

    delete[] D;
    delete[] Q;
}


template<class T, bool IS_COL_MAJOR>
static void generateRandomPDMat( T* A, const int dim, const T cond_num, std::default_random_engine& e )
{
    std::uniform_real_distribution<T> dist_positive {  1.001,  cond_num };
    std::uniform_real_distribution<T> dist_any      { -0.9999, 0.9999   };

    T* D = new T[ dim ];
    T* Q = new T[ dim * dim ];

    for ( int i = 0; i < dim; i++ ) {

        D[i] = dist_positive( e );
    }

    for ( int i = 0; i < dim; i++ ) {

        for ( int j = 0; j <= i; j++ ) {

            const T val = dist_any( e );

            Q  [ linear_index_mat<IS_COL_MAJOR>(i, j, dim, dim) ] = val;
        }
    }

    for ( int i = 0 ; i < dim ; i++ ) {

        for ( int j = 0; j <= i; j++ ) {

            T sum = 0.0;

            for ( int k = 0; k < dim; k++ ) {

                const T Q_ik  = (i>=k) ? Q[ linear_index_mat<IS_COL_MAJOR>( i, k, dim, dim ) ] : Q[ linear_index_mat<IS_COL_MAJOR>( k , i, dim, dim ) ];
                const T Qt_kj = (j>=k) ? Q[ linear_index_mat<IS_COL_MAJOR>( j, k, dim, dim ) ] : Q[ linear_index_mat<IS_COL_MAJOR>( k , j, dim, dim ) ];

                sum += ( Q_ik * Qt_kj );
            }

            A [ linear_index_mat<IS_COL_MAJOR>( i , j, dim, dim ) ] = sum;
            A [ linear_index_mat<IS_COL_MAJOR>( j , i, dim, dim ) ] = sum;
        }
    }

    // check and correct strict diagonal dominance.
    // add diagonal elements.
    for ( int i = 0 ; i < dim ; i++ ) {

        T sum = 0.0;
        for ( int j = 0; j < dim; j++ ) {
            if ( j != i ) {
                sum += abs(( A[ linear_index_mat<IS_COL_MAJOR>( i , j, dim, dim) ] ));
            }
        }

        const T diff = A[ linear_index_mat<IS_COL_MAJOR>( i , i, dim, dim ) ] - sum;

        if ( diff < 0.0 ) {

            //std::cerr << "A[" << i << "," << i << "] too small. Correcting" << diff << "\n";

            A[ linear_index_mat<IS_COL_MAJOR>( i , i, dim, dim ) ] -= diff;
        }

        A[ linear_index_mat<IS_COL_MAJOR>( i, i, dim, dim ) ] *= D[i];
    }

    delete[] D;
    delete[] Q;
}



template<class T>
static void generateRandomTimeVector512( T* re, T*im, const T max_amp, const int num_sines, std::default_random_engine& e )
{
    uniform_real_distribution<T>  dist_amp   { 1.0, max_amp };
    uniform_real_distribution<T>  dist_phase { -1.0* M_PI, M_PI };

    for ( int i = 0; i < num_sines; i++ ) {

        T amp   = dist_amp   ( e );
        T phase = dist_phase ( e );

        for ( int j = 0; j < 512; j++ ) {

            T rad = (T)i * (T)j / 512.0 + phase;

            re[j] = amp * cos( rad );
            im[j] = amp * sin( rad );
        }
    }
}



#endif /*__TEST_PATTERN_GENERATION_H__*/

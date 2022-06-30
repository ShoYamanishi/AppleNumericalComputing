#ifndef __TEST_CASE_LCP_LEMKE_BASELINE_H__
#define __TEST_CASE_LCP_LEMKE_BASELINE_H__
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <Accelerate/Accelerate.h>

#include "test_case_lcp.h"


template<class T, bool IS_COL_MAJOR>
class TestCaseLCP_lemke_baseline : public TestCaseLCP<T, IS_COL_MAJOR> {

    // Type of problem that can be solved:
    //
    //   M z + q = w
    //
    //   s.t. 0 <= z cmpl. w >= 0
    //
    //   M must not be symmetric ( For the precise list of the types of matrices accepted.
    //   please see 2.3 "CONDITIONS UNDER WHICH THE COMPLEMENTARY PIVOT ALGORITHM WORKS"
    //   of "LINEAR COMPLEMENTARITY LINEAR AND NONLINEAR PROGRAMMING" by Katta G. Murty.)
    //
    // - All the constraints must be unilateral constraints, i.e., 0<=z cmpl. w>=0.
    // 
    // - Boxed contraints can be solved with the following reformulation of the problem.
    //
    //   e.g.
    //
    //       |  A    B  |   |Zb|   |qb|    |Wb|
    //   M = |          | x |  | + |  | >= |  |
    //       |  C    D  |   |Zu|   |qz|    |Wu|
    //
    //   s.t.  lo_i < Zb_i < hi_i <=> Wb_i == 0
    //            lo_i = Zb_i     <=> Wb_i > 0
    //            hi_i = Zb_i     <=> Wb_i < 0
    //
    //         0 <= Zu cmpl. Wu >= 0
    //
    //   Expand the equations as follows
    //
    //       |  A  -A   B  |   |Zb1|   |qb - lo|    |Wb1|
    //       |             |   |   |   |       |    |   |
    //   M = | -A   A  -B  | x |Zb2| + |hi - qb| >= |Wb2|
    //       |             |   |   |   |       |    |   |
    //       |  C  -C   D  |   |Zu |   |  qz   |    |Wu |
    //   
    //   s.t.  0 <= Zb1 cmpl. Wb1 >= 0
    //         0 <= Zb2 cmpl. Wb2 >= 0
    //         0 <= Zu  cmpl. Wu  >= 0
    //
    //         Zb = Zb1 - Zb2
    //
    // - Mixed complementarity problem with some bilateral constraints can be solved as follows.
    //   e.g. 
    //       |  A    B  |   |Zb|   |qb| = |0 |
    //   M = |          | x |  | + |  |   |  |
    //       |  C    D  |   |Zu|   |qz|>= |Wu|
    //
    //   This can be decomposed into two parts: bilateral and unilateral as follows.
    //
    //   - A x Zb + B x Zu + qb = 0  (bilateral) - (1)
    // 
    //   - C x Zb + D x Zu + qz >= Wu (unilateral) - (2)
    //
    //                    -1            -1
    //   from (1)  Zb = -A  x B x Zu - A  x qb  - (3)
    //
    //                            -1                      -1
    //   from (2) and (3) ( -C x A  x B + D ) x Zu - C x A  x qb + qz >= Wu - (4)
    // 
    //   First, make the inverse of A (e.g. by cholesky decomposition if A is symmetric)
    //   Then solve (4) with this Lemke solver for Wu and Zu.
    //   Finally use (3) to solve Zb from Zu.
    //

    // Description of the table
    // ------------------------
    // 
    // width: |    m_dim    |    m_dim    |       1     |       1     |
    // -------+-------------+-------------+-------------+-------------+
    //        |             |             |             |             |
    // field  |     -1      |     -1      |     -1      |      -1     |
    // type   |    B   I    |    B   M    |    B   e    |     B  q    |
    //        |             |             |             |             |
    // -------+-------------+-------------+-------------+-------------+
    //        |             |             |             |             |
    //        | ^^^^^^^^^^^ |
    // desc-  | Columns for the slack variables. 
    // ription| The original columns constitute the identity matrix,
    //        | and the up-to-date columns here represent the current
    //        | inverse for the matrix B for the basic variables.
    //        |             |
    //        |             | ^^^^^^^^^^^
    //        |             | Columns for the original variables z.
    //        |             | The original columns correspond to the matrix M.
    //        |             |             |
    //        |             |             | ^^^^^^^^^^^ |
    //        |             |             | Column for the artificial varialble z_0
    //        |             |             | It originally consists of -1s.
    //        |             |             |             |             |
    //        |             |             |             | ^^^^^^^^^^^ |
    //        |             |             |             | The column for RHS q.

    T* m_table;

    // array of indices for the rows of the table to keep track of the current basic variables.
    int*   m_basic_variables_along_rows;

    const int    m_num_columns;
    const int    m_z0_index;
    const int    m_max_num_iterations;

  public:

    TestCaseLCP_lemke_baseline( const int dim, const T condition_num, const int max_num_iterations, const T epsilon, const LCPTestPatternType p_type )
        :TestCaseLCP<T, IS_COL_MAJOR>( dim, condition_num, epsilon, p_type )
        ,m_num_columns        ( dim + dim + 1 + 1 )
        ,m_z0_index           ( dim + dim )
        ,m_max_num_iterations ( max_num_iterations )
    {
        if constexpr (IS_COL_MAJOR) {
            assert(true); //column major not supported for Lemke.
        }
        m_table           = new T [ dim * (m_num_columns ) ];
        m_basic_variables_along_rows = new int   [ dim ];

        this->setImplementationType( LEMKE_CPP );
    }

    virtual ~TestCaseLCP_lemke_baseline(){

        delete[] m_table;
        delete[] m_basic_variables_along_rows;
    }

    virtual void run() {

        int num_iterations = 0;

        if ( find_min_q() >= 0.0 ) {
            // already feasible.
            memset( this->m_z,         0, sizeof(T) * this->m_dim );
            memcpy( this->m_w, this->m_q, sizeof(T) * this->m_dim );
            return;
        }

        // from here on we assume m_q has at least one negative element.

        fill_initial_values();

        const auto initial_entering_col_index = this->m_dim + this->m_dim; //z_0
        const auto initial_leaving_row_index  = find_initial_leaving_row_index();

        pivot( initial_entering_col_index, initial_leaving_row_index );

        int entering_col_index = get_complementary_index( initial_leaving_row_index );

        m_basic_variables_along_rows[initial_leaving_row_index] = initial_entering_col_index;

        // pivot loop until z_0 becomes non-basic
        while ( num_iterations++ < m_max_num_iterations && entering_col_index != m_z0_index ) {

            int leaving_row_index = find_leaving_row_index( entering_col_index );

            if ( leaving_row_index == -1 ) {
                cerr << "ERROR: ray-termination\n";
                break; // ray-termination
            }

            pivot( entering_col_index, leaving_row_index );

            int new_entering_col_index = get_complementary_index( leaving_row_index );

            m_basic_variables_along_rows[leaving_row_index] = entering_col_index;

            entering_col_index = new_entering_col_index;
        }        

        if ( num_iterations >=  m_max_num_iterations ) {
            cerr << "ERROR: max numbere of pivoting exceeded.\n";
        }

        // arrange solutions in m_z and m_w.
        memset( this->m_z, 0, sizeof(T) * this->m_dim );
        memset( this->m_w, 0, sizeof(T) * this->m_dim );

        for ( int row_index = 0; row_index < this->m_dim; row_index++ ) {

            auto col_index = m_basic_variables_along_rows[row_index];

            if ( col_index < this->m_dim ) {

                // slack variable w_i
                this->m_w[col_index] = this->m_table[ row_index * m_num_columns + this->m_dim + this->m_dim + 1 ];
            }
            else if ( col_index < m_z0_index ) {

                // real variable z_i
                this->m_z[ col_index - this->m_dim ] = this->m_table[ row_index * m_num_columns + this->m_dim + this->m_dim + 1 ];
            }
            else {

                cerr << "ERROR: final solution contains z0\n";
            }

        }
        this->setIterations( num_iterations, 0, 0 );
    }

    inline T find_min_q() {
        T min_q = this->m_q[0];
        for ( int i = 1; i < this->m_dim; i++ ) {
            min_q = min( min_q, this->m_q[i] );
        }
        return min_q;
    }

    int get_complementary_index( const int leaving_row ) {

        const int col_index =  m_basic_variables_along_rows[leaving_row];

        if ( col_index < this->m_dim ) { // index is for a slack varible w_i

            return col_index + this->m_dim; // return z_i
        }
        else if ( col_index < m_z0_index ) { // index is for a real variable z_i

            return col_index - this->m_dim; // return the slack variable w_i
        }
        else { // index is the artificial variable z_0.

            return m_z0_index;
        }
    }

    void  pivot( const int entering_col_index, const int leaving_row_index ) {

        // the body of this for loop can be split into blocks for multiple threads.

        for ( int row_index = 0; row_index < this->m_dim; row_index++ ) {

            T* pivot_row   = &(m_table[ leaving_row_index * m_num_columns ]);

            const T pivot_denom = 1.0 / pivot_row[ entering_col_index ];

            if ( row_index != leaving_row_index ) {

                T* current_row = &(m_table[ row_index * m_num_columns ]);

                const T coeff = -1.0 * current_row[ entering_col_index ] * pivot_denom;

                for ( int col_index = 0; col_index < this->m_num_columns; col_index++ ) {
                    current_row[col_index] += (pivot_row[col_index] * coeff);
                }

                current_row[ entering_col_index ] = 0.0;
            }
            else {

                T*       pivot_row = &(m_table[ leaving_row_index * m_num_columns ]);

                for ( int col_index = 0; col_index < this->m_num_columns; col_index++ ) {
                    pivot_row[col_index] *= pivot_denom;
                }

                pivot_row[ entering_col_index ] = 1.0;
            }
        }
    }

    void fill_initial_values() {

        for ( int row_index = 0; row_index < this->m_dim; row_index++ ) {

            // Slack part. Fill with 0.0 first.
            memset( &(m_table[ row_index * m_num_columns ]), 0, sizeof(T) * this->m_dim );

            // Slack part diagonal with 1.0
            m_table[ row_index * m_num_columns + row_index ] = 1.0;


            // -M part. m_table[ row ] := m_M[ row ] * -1.0
            for ( int col_index = 0; col_index < this->m_dim; col_index++ ) {

                m_table[ row_index * m_num_columns + this->m_dim + col_index ]
                    = -1.0 * this->m_M[ row_index * this->m_dim + col_index ];
            }

            // z_0 part. Fill with -1.0.
            m_table[ row_index * m_num_columns + this->m_dim + this->m_dim ] = -1.0;

            // q part.
            m_table[ row_index * m_num_columns + this->m_dim + this->m_dim + 1 ] = this->m_q[ row_index ];

            // initially, all the rows correspond to the slack variables.
            m_basic_variables_along_rows[row_index] = row_index;
        }
    }

    int find_initial_leaving_row_index() {

        // find the minimum row.
        // It's not necessarity the lexico minimum,
        // but since it happens only once in the beginning,
        // it does not cause a pivot cycle.
        T min_q = this->m_q[0];
        unsigned long min_q_index = 0;

        for ( int i = 1; i < this->m_dim; i++ ) {

            if ( min_q > this->m_q[i] ) {
                min_q = this->m_q[i];
                min_q_index = i;
            }
        }

        return (int)min_q_index;
    }

    // Find the lexico minimum 
    int find_leaving_row_index(int entering_table_index) {

        std::vector<int> active_rows;        
        T current_min = std::numeric_limits<T>::max();

        for ( int row_index = 0; row_index < this->m_dim; row_index++ ) {

            const T denom = m_table[ row_index * m_num_columns + entering_table_index ];

            if ( denom > this->m_epsilon ) {

                const T q = m_table[ row_index * m_num_columns + this->m_dim + this->m_dim + 1 ] / denom;

                if ( fabs(current_min - q) < this->m_epsilon ) {

                    active_rows.push_back(row_index);                    
                }                
                else if ( current_min > q ) {
                    current_min = q;
                    active_rows.clear();
                    active_rows.push_back(row_index);
                }
            }
        }

        if ( active_rows.size() == 0 ) {
            // ray termination.
            return -1;
        }
        else if ( active_rows.size() == 1 ) {

            return *(active_rows.begin());
        }

        // If there are multiple rows, check if they contain the row for z_0.
        for (auto it = active_rows.begin(); it !=  active_rows.end(); it++ ) {
            if ( m_basic_variables_along_rows[*it] == m_z0_index ) {
                return *it;
            }
        }

        // look through the columns of the inverse of the basic matrix from left to right.
        // until the tie is broken.
        for ( int col_index = 0; col_index < this->m_dim ; col_index++ ) {

            std::vector<int> active_rows_copy = active_rows;
            active_rows.clear();

            current_min = std::numeric_limits<T>::max();

            for ( auto it = active_rows_copy.begin(); it != active_rows_copy.end(); it++ ) {
                const int row_index = *it;

                // denom must be positive here.
                const T denom = m_table[ row_index * m_num_columns + entering_table_index ];

                const T b_col = m_table[ row_index * m_num_columns + col_index ] / denom;

                if ( fabs(current_min - b_col) < this->m_epsilon ) {
                    active_rows.push_back(row_index);
                }                
                else if ( current_min > b_col ) {
                    current_min = b_col;
                    active_rows.clear();
                    active_rows.push_back(row_index);
                }
            }

            if ( active_rows.size() == 1 ) {
                return *(active_rows.begin());
            }
        }

        assert(true);// must not reach here.
        
        return -1;
    }

    string from_col_index_to_var_name(const int i) {

        if ( i < this->m_dim ) {

            return  "w" + std::to_string( i + 1 );
        }
        else if  ( i < this->m_z0_index ) {

            return "z" + std::to_string( i + 1 - this->m_dim );
        }
        else if ( i == this->m_z0_index ) {
            return "z0";
        }
        else {
            return "q";
        }
    }

    void print_table(ostream& os) {

        os << setprecision(3);
        os << "\n\t";
        for ( int col_index = 0; col_index < m_num_columns; col_index++ ) {
            os << from_col_index_to_var_name(col_index) << "\t";
        }
        os << "\n";

        for ( int row_index = 0; row_index < this->m_dim; row_index++ ) {
            int col_index = m_basic_variables_along_rows[row_index];

            os << from_col_index_to_var_name(col_index) << "\t";

            for ( int col_index = 0; col_index < m_num_columns; col_index++ ) {
                os << m_table[row_index * this->m_num_columns + col_index] << "\t";
            }
            os << "\n";
        }
    }

    void print_enter_leave( ostream& os, const int entering_col_index , const int leaving_row_index ) {

        os << "entering_col_index: " << entering_col_index << " "
            << from_col_index_to_var_name(entering_col_index) << "\n";

        os << "leaving_row_index: " << leaving_row_index << " "
            << from_col_index_to_var_name( m_basic_variables_along_rows[leaving_row_index]) << "\n";
    }

};

#endif /*__TEST_CASE_LCP_LEMKE_BASELINE_H__*/

#ifndef __TEST_LCP_PATTERN_GENERATOR_H__
#define __TEST_LCP_PATTERN_GENERATOR_H__

#include <string>
#include <fstream>
#include <algorithm>
#include "test_pattern_generation.h"

const std::string SAMPLE_DATA_32_MU02   = "sample_data_32_mu0.2.txt";
const std::string SAMPLE_DATA_64_MU02   = "sample_data_64_mu0.2.txt";
const std::string SAMPLE_DATA_128_MU02  = "sample_data_128_mu0.2.txt";
const std::string SAMPLE_DATA_256_MU02  = "sample_data_256_mu0.2.txt";
const std::string SAMPLE_DATA_512_MU02  = "sample_data_512_mu0.2.txt";
const std::string SAMPLE_DATA_1024_MU02 = "sample_data_1024_mu0.2.txt";

const std::string SAMPLE_DATA_32_MU08   = "sample_data_32_mu0.8.txt";
const std::string SAMPLE_DATA_64_MU08   = "sample_data_64_mu0.8.txt";
const std::string SAMPLE_DATA_128_MU08  = "sample_data_128_mu0.8.txt";
const std::string SAMPLE_DATA_256_MU08  = "sample_data_256_mu0.8.txt";
const std::string SAMPLE_DATA_512_MU08  = "sample_data_512_mu0.8.txt";
const std::string SAMPLE_DATA_1024_MU08 = "sample_data_1024_mu0.8.txt";

const std::string SAMPLE_DATA_32_SYM    = "sample_data_32_sym.txt";
const std::string SAMPLE_DATA_64_SYM    = "sample_data_64_sym.txt";
const std::string SAMPLE_DATA_128_SYM   = "sample_data_128_sym.txt";
const std::string SAMPLE_DATA_256_SYM   = "sample_data_256_sym.txt";
const std::string SAMPLE_DATA_512_SYM   = "sample_data_512_sym.txt";
const std::string SAMPLE_DATA_1024_SYM  = "sample_data_1024_sym.txt";

typedef enum _LCPTestPatternType {

    LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC,
    LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC,
    LCP_REAL_NONSYMMETRIC_MU02,
    LCP_REAL_NONSYMMETRIC_MU08,
    LCP_REAL_SYMMETRIC

} LCPTestPatternType;


template <class T, bool IS_COL_MAJOR>
class LCPPatternGenerator {

  protected:

    default_random_engine m_e;
    const T               m_val_low_q;
    const T               m_val_high_q;
    const std::string     m_test_pattern_path;

  public:

    LCPPatternGenerator(const bool repeatable, const T val_low_q, const T val_high_q, const std::string test_pattern_path )
        :m_e         (   repeatable 
                       ? 0 
                       : chrono::system_clock::now().time_since_epoch().count()
                     )
        ,m_val_low_q        ( val_low_q )
        ,m_val_high_q       ( val_high_q )
        ,m_test_pattern_path( test_pattern_path )
        {}


    ~LCPPatternGenerator(){}

    void generateTestPattern(
        const int                dim,
        T*                       M,
        T*                       q,
        const LCPTestPatternType p_type,
        const T                  condition_num
    ) {

        if ( p_type == LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC ) {

            generateRandomCopositiveMat<T>( M, dim, condition_num, m_e );

            fillArrayWithRandomValues<T>( m_e, q, dim, m_val_low_q, m_val_high_q );
        }
        else if ( p_type == LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC ) {

            generateRandomPDMat<T, IS_COL_MAJOR>( M, dim, condition_num, m_e );

            fillArrayWithRandomValues<T>( m_e, q, dim, m_val_low_q, m_val_high_q );
        }
        else if ( p_type == LCP_REAL_NONSYMMETRIC_MU02 ) { 

            std::string file_name;

            if ( dim <= 32 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_32_MU02;
            }
            else if ( dim <= 64 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_64_MU02;
            }
            else if ( dim <= 128 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_128_MU02;
            }
            else if ( dim <= 256 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_256_MU02;
            }
            else if ( dim <= 512 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_512_MU02;
            }
            else if ( dim <= 1024 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_1024_MU02;
            }
            else {
                assert(true);
            }

            load_test_data( file_name, dim, M, q );
        }
        else if ( p_type == LCP_REAL_NONSYMMETRIC_MU08 ) { 

            std::string file_name;

            if ( dim <= 32 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_32_MU08;
            }
            else if ( dim <= 64 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_64_MU08;
            }
            else if ( dim <= 128 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_128_MU08;
            }
            else if ( dim <= 256 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_256_MU08;
            }
            else if ( dim <= 512 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_512_MU08;
            }
            else if ( dim <= 1024 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_1024_MU08;
            }
            else {
                assert(true);
            }

            load_test_data( file_name, dim, M, q );
        }
        else if ( p_type == LCP_REAL_SYMMETRIC ) { 

            std::string file_name;

            if ( dim <= 32 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_32_SYM;
            }
            else if ( dim <= 64 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_64_SYM;
            }
            else if ( dim <= 128 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_128_SYM;
            }
            else if ( dim <= 256 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_256_SYM;
            }
            else if ( dim <= 512 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_512_SYM;
            }
            else if ( dim <= 1024 ) {
                file_name = m_test_pattern_path + SAMPLE_DATA_1024_SYM;
            }
            else {
                assert(true);
            }

            load_test_data( file_name, dim, M, q );
        }
    }

    void load_test_data( const std::string& file_name, const int dim, T* M, T* q ) {

        std::ifstream file;

        memset(M, 0, sizeof(T) * dim * dim );
        memset(q, 0, sizeof(T) * dim );

        file.open( file_name );

        if ( file.is_open() ) {

            std::string line;
            while (std::getline(file, line)) {
                line.erase(std::remove_if( line.begin(), line.end(), [](auto ch){ return (ch == '\n' || ch == '\r'); }),line.end() );

                vector<std::string> tokens = tokenize( line, "\t" );
                if ( tokens.size() == 3 ) {
                    if ( tokens[0][0] == 'q' ) {
                        const int i = atoi(tokens[1].c_str());
                        const T v   = atof(tokens[2].c_str());
                        if ( 0 <= i && i < dim ) {
                            q[i] = v;
                        }
                    }
                    else {
                        const int row = atoi(tokens[0].c_str());
                        const int col = atoi(tokens[1].c_str());
                        const T v     = atof(tokens[2].c_str());
                        if ( 0 <= row && row < dim &&  0 <= col && col < dim  ) {
                            M[ row*dim + col ] = v;
                        }
                    }
                }
            }
            file.close();
        }

//        cerr << setprecision(2);
//        for (int row = 0; row < dim ; row++ ) {
//            cerr << "row[" << row << "]";
//            for (int col = 0; col < dim ; col++ ) {
//                cerr << "\t" << M[row*dim + col];
//            }
//            cerr << "\tq:" << q[row] << "\n";
//        }
    }

    vector<std::string> tokenize( std::string line, const std::string delimiter ){

        vector<std::string> tokens;

        size_t pos = 0;
        std::string token;

        while ( ( pos = line.find( delimiter ) ) != std::string::npos ) {

            token = line.substr( 0, pos );
            tokens.push_back( token );
            line.erase( 0, pos + delimiter.length() );

        }
        tokens.push_back( line );
        return tokens;
    }

};

#endif /*__TEST_LCP_PATTERN_GENERATOR_H__*/

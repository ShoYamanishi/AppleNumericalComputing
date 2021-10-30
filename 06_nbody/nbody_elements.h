#ifndef __NBODY_ELEMENTS_H__
#define __NBODY_ELEMENTS_H__
#include "test_case_with_time_measurements.h"

template<class T>
class VelocityElem {
  public:
    VelocityElem(){ static_type_guard_real<T>(); }
    VelocityElem<T>& operator = ( const VelocityElem<T>& rhs ); 
};


template<class T>
class NBodyElem {

  public:

    NBodyElem(){ static_type_guard_real<T>(); }

    void setRandomInitialState( default_random_engine& e );

    NBodyElem<T>& operator=( const NBodyElem<T>& rhs ); 
};


template<class T>
class NBodySOA {

  public:

    size_t m_num_elements;
    bool   m_p0_is_active;

    T* m_p0x;
    T* m_p0y;
    T* m_p0z;

    T* m_p1x;
    T* m_p1y;
    T* m_p1z;

    T* m_vx;
    T* m_vy;
    T* m_vz;

    T* m_ax;
    T* m_ay;
    T* m_az;

    T* m_mass;



    NBodySOA( size_t num_elements )

        : m_num_elements( num_elements )

         ,m_p0_is_active( true )
    {
        static_type_guard_real<T>();

        m_p0x  = new T[ num_elements ];
        m_p0y  = new T[ num_elements ];
        m_p0z  = new T[ num_elements ];

        m_p1x  = new T[ num_elements ];
        m_p1y  = new T[ num_elements ];
        m_p1z  = new T[ num_elements ];

        m_vx   = new T[ num_elements ];
        m_vy   = new T[ num_elements ];
        m_vz   = new T[ num_elements ];

        m_ax   = new T[ num_elements ];
        m_ay   = new T[ num_elements ];
        m_az   = new T[ num_elements ];

        m_mass = new T[ num_elements ];
    }

    ~NBodySOA()
    {
        delete[] m_p0x;
        delete[] m_p0y;
        delete[] m_p0z;

        delete[] m_p1x;
        delete[] m_p1y;
        delete[] m_p1z;

        delete[] m_vx;
        delete[] m_vy;
        delete[] m_vz;

        delete[] m_ax;
        delete[] m_ay;
        delete[] m_az;

        delete[] m_mass;
    }

    void set( const NBodyElem<T>& e, const size_t i ) {

        m_p0x[i] = e.m_p0.x;
        m_p0y[i] = e.m_p0.y;
        m_p0z[i] = e.m_p0.z;

        m_p1x[i] = e.m_p1.x;
        m_p1y[i] = e.m_p1.y;
        m_p1z[i] = e.m_p1.z;

        m_vx[i] = e.m_v.x;
        m_vy[i] = e.m_v.y;
        m_vz[i] = e.m_v.z;

        m_ax[i] = e.m_am.x;
        m_ay[i] = e.m_am.y;
        m_az[i] = e.m_am.z;

        m_mass[i] = e.m_am.w;
    }

    void get( NBodyElem<T>& e, const size_t i ) const {

        e.m_p0.x = m_p0x[i];
        e.m_p0.y = m_p0y[i];
        e.m_p0.z = m_p0z[i];

        e.m_p1.x = m_p1x[i];
        e.m_p1.y = m_p1y[i];
        e.m_p1.z = m_p1z[i];

        e.m_v.x = m_vx[i];
        e.m_v.y = m_vy[i];
        e.m_v.z = m_vz[i];

        e.m_am.x = m_ax[i];
        e.m_am.y = m_ay[i];
        e.m_am.z = m_az[i];

        e.m_am.w = m_mass[i];
    }

};
#endif /*__NBODY_ELEMENTS_H__*/

#ifndef __NBODY_ELEMENTS_IMPL_H__
#define __NBODY_ELEMENTS_IMPL_H__

#include <Accelerate/Accelerate.h>
#include <simd/simd.h>
#include "nbody_elements.h"

template<>
class VelocityElem<float> {
  public:

    simd_float3 m_v ;

    VelocityElem<float>& operator=( const VelocityElem<float>& rhs ) {
        m_v  = rhs.m_v;
        return *this;
    }

    VelocityElem<float>& operator=( const simd_float3& rhs ) { 
        m_v  = rhs;
        return *this;
    }
};

template<>
class VelocityElem<double> {

  public:

    simd_double3 m_v ;

    VelocityElem<double>& operator=( const VelocityElem<double>& rhs ) { 
        m_v  = rhs.m_v;
        return *this;
    }

    VelocityElem<double>& operator=( const simd_double3& rhs ) { 
        m_v  = rhs;
        return *this;
    }
};


template<>
class NBodyElem<float> {

  public:
    simd_float3  m_p0 ;
    simd_float3  m_p1 ;
    simd_float3  m_v  ;
    simd_float4  m_am ; // xyz: accel, w: mass

    void setRandomInitialState( default_random_engine& e ) {

        uniform_real_distribution dist_geom{ -1.0, 1.0 };
        uniform_real_distribution dist_mass{  0.0, 1.0 };

        m_p0.x = dist_geom( e );
        m_p0.y = dist_geom( e );
        m_p0.z = dist_geom( e );

        m_p1.x = dist_geom( e );
        m_p1.y = dist_geom( e );
        m_p1.z = dist_geom( e );

        m_v.x = dist_geom( e );
        m_v.y = dist_geom( e );
        m_v.z = dist_geom( e );

        m_am.x = 0.0;
        m_am.y = 0.0;
        m_am.z = 0.0;
        m_am.w = dist_mass( e );
    }

    NBodyElem<float>& operator=( const NBodyElem<float>& rhs ) {

        m_p0 = rhs.m_p0;
        m_p1 = rhs.m_p1;
        m_v  = rhs.m_v;
        m_am = rhs.m_am;

        return *this;
    }

    bool operator!=( const NBodyElem<float>& rhs ) {

        return    m_p0.x != rhs.m_p0.x
               || m_p0.y != rhs.m_p0.y
               || m_p0.z != rhs.m_p0.z

               || m_p1.x != rhs.m_p1.x
               || m_p1.y != rhs.m_p1.y
               || m_p1.z != rhs.m_p1.z

               || m_v.x != rhs.m_v.x
               || m_v.y != rhs.m_v.y
               || m_v.z != rhs.m_v.z

               || m_am.x != rhs.m_am.x
               || m_am.y != rhs.m_am.y
               || m_am.z != rhs.m_am.z
               || m_am.w != rhs.m_am.w; 
    }

    bool equalWithinTolerance( const NBodyElem<float>& rhs, const float tolerance ) {
/*
        return    ::equalWithinTolerance( m_p0.x, rhs.m_p0.x, tolerance )
               && ::equalWithinTolerance( m_p0.y, rhs.m_p0.y, tolerance )
               && ::equalWithinTolerance( m_p0.z, rhs.m_p0.z, tolerance )

               && ::equalWithinTolerance( m_p1.x, rhs.m_p1.x, tolerance )
               && ::equalWithinTolerance( m_p1.y, rhs.m_p1.y, tolerance )
               && ::equalWithinTolerance( m_p1.z, rhs.m_p1.z, tolerance )

               && ::equalWithinTolerance( m_v.x,  rhs.m_v.x,  tolerance )
               && ::equalWithinTolerance( m_v.y,  rhs.m_v.y,  tolerance )
               && ::equalWithinTolerance( m_v.z,  rhs.m_v.z,  tolerance )

               && ::equalWithinTolerance( m_am.x, rhs.m_am.x, tolerance )
               && ::equalWithinTolerance( m_am.y, rhs.m_am.y, tolerance )
               && ::equalWithinTolerance( m_am.z, rhs.m_am.z, tolerance )
               && ::equalWithinTolerance( m_am.w, rhs.m_am.w, tolerance )
               ;
*/
        return    ::equalWithinTolerance( m_am.x, rhs.m_am.x, tolerance )
               && ::equalWithinTolerance( m_am.y, rhs.m_am.y, tolerance )
               && ::equalWithinTolerance( m_am.z, rhs.m_am.z, tolerance )
               && ::equalWithinTolerance( m_am.w, rhs.m_am.w, tolerance )
               ;
    }
};

template<>
class NBodyElem<double> {
  public:
    simd_double3  m_p0 ;
    simd_double3  m_p1 ;
    simd_double3  m_v  ;
    simd_double4  m_am ; // xyz: accel, w: mass

    void setRandomInitialState( default_random_engine& e ) {

        uniform_real_distribution dist_geom{ -1.0, 1.0 };
        uniform_real_distribution dist_mass{  0.0, 1.0 };

        m_p0.x = dist_geom( e );
        m_p0.y = dist_geom( e );
        m_p0.z = dist_geom( e );

        m_p1.x = dist_geom( e );
        m_p1.y = dist_geom( e );
        m_p1.z = dist_geom( e );

        m_v.x = dist_geom( e );
        m_v.y = dist_geom( e );
        m_v.z = dist_geom( e );

        m_am.x = 0.0;
        m_am.y = 0.0;
        m_am.z = 0.0;
        m_am.w = dist_mass( e );
    }

    NBodyElem<double>& operator=( const NBodyElem<double>& rhs ) {

        m_p0.x = rhs.m_p0.x;
        m_p0.y = rhs.m_p0.y;
        m_p0.z = rhs.m_p0.z;

        m_p1.x = rhs.m_p1.x;
        m_p1.y = rhs.m_p1.y;
        m_p1.z = rhs.m_p1.z;

        m_v.x = rhs.m_v.x;
        m_v.y = rhs.m_v.y;
        m_v.z = rhs.m_v.z;

        m_am.x = rhs.m_am.x;
        m_am.y = rhs.m_am.y;
        m_am.z = rhs.m_am.z;
        m_am.w = rhs.m_am.w; 

        return *this;
    }

    bool operator!=( const NBodyElem<double>& rhs ) {

        return    m_p0.x != rhs.m_p0.x
               || m_p0.y != rhs.m_p0.y
               || m_p0.z != rhs.m_p0.z

               || m_p1.x != rhs.m_p1.x
               || m_p1.y != rhs.m_p1.y
               || m_p1.z != rhs.m_p1.z

               || m_v.x  != rhs.m_v.x
               || m_v.y  != rhs.m_v.y
               || m_v.z  != rhs.m_v.z

               || m_am.x != rhs.m_am.x
               || m_am.y != rhs.m_am.y
               || m_am.z != rhs.m_am.z
               || m_am.w != rhs.m_am.w; 
    }

    bool equalWithinTolerance( const NBodyElem<double>& rhs, const double tolerance ) {

        return    ::equalWithinTolerance( m_p0.x, rhs.m_p0.x, tolerance )
               && ::equalWithinTolerance( m_p0.y, rhs.m_p0.y, tolerance )
               && ::equalWithinTolerance( m_p0.z, rhs.m_p0.z, tolerance )

               && ::equalWithinTolerance( m_p1.x, rhs.m_p1.x, tolerance )
               && ::equalWithinTolerance( m_p1.y, rhs.m_p1.y, tolerance )
               && ::equalWithinTolerance( m_p1.z, rhs.m_p1.z, tolerance )

               && ::equalWithinTolerance( m_v.x,  rhs.m_v.x,  tolerance )
               && ::equalWithinTolerance( m_v.y,  rhs.m_v.y,  tolerance )
               && ::equalWithinTolerance( m_v.z,  rhs.m_v.z,  tolerance )

               && ::equalWithinTolerance( m_am.x, rhs.m_am.x, tolerance )
               && ::equalWithinTolerance( m_am.y, rhs.m_am.y, tolerance )
               && ::equalWithinTolerance( m_am.z, rhs.m_am.z, tolerance )
               && ::equalWithinTolerance( m_am.w, rhs.m_am.w, tolerance )
               ;
    }
};

#endif /*__NBODY_ELEMENTS_IMPL_H__*/

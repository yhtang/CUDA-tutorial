#ifndef MATH_H_
#define MATH_H_

#include <cuda.h>

namespace cuda {

// SIN
template<typename T>
__inline__ __device__ T sin( T x ) { return ::sin(x); }

template<>
__inline__ __device__ float sin( float x ) { return ::sinf(x); }

#define Overload(fun) \
		template<typename T> \
		__inline__ __device__ T fun( T x ) { return ::fun(x); } \
		template<> \
		__inline__ __device__ float fun( float x ) { return ::fun##f(x); }

Overload(cos);
Overload(exp);
Overload(log);
Overload(tan);

#undef Overload

template<typename T>
struct literal {
    T value;
    literal( T v = 0 ) : value(v) {}
    __inline__ __host__ __device__ operator T& () {
        return value;
    }
};

}


#endif

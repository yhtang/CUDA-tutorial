#ifndef UTIL_H_
#define UTIL_H_

#ifdef __CUDACC__

__device__ int block_id() {
	return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ int block_num() {
	return gridDim.x * gridDim.y * gridDim.z;
}

__device__ int block_size() {
	return blockDim.x * blockDim.y * blockDim.z;
}

__device__ int local_thread_id() {
	return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__device__ int global_thread_id() {
	return block_id() * block_size() + local_thread_id();
}

__device__ int global_thread_num() {
	return block_num() * block_size();
}

#endif

#include <ctime>
#include <cstdlib>

double get_time()
{
	struct timespec time;
	clock_gettime( CLOCK_REALTIME, &time );
	return (double)time.tv_sec + (double)time.tv_nsec * 1.0e-9 ;
}

#endif

#ifndef CHECK_CUDA_ERROR_H
#define CHECK_CUDA_ERROR_H

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) { \
 cudaError_t err; \
 if ( (err = (call)) != cudaSuccess) { \
 fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
 __FILE__, __LINE__); \
 exit(1); \
 } \
} 

#endif // CHECK_CUDA_ERROR_H

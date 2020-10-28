#pragma once


#include <helper_math.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

#define SIGNAL_ERROR( msg ) \
    printf(msg); \
    exit(EXIT_FAILURE);


inline static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
inline void __getLastCudaError(const char* errorMessage, const char* file, const int line)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
// This will output the proper error string when calling cudaGetLastError
#define CHECK_CUDA_ERROR(msg) __getLastCudaError (msg, __FILE__, __LINE__)

#define CHECK_IF_CUDA_ERROR(msg) CHECK_CUDA_ERROR(msg);printf("NO ERROR AT %s \n",msg);

#define HANDLE_ERROR( err ) \
    cudaDeviceSynchronize(); \
    HandleError( err, __FILE__, __LINE__ )


#define HANDLE_NULL( a ) {\
	if (a == NULL) { \
		printf( "Host memory failed in %s at line %d\n",  __FILE__, __LINE__ ); \
		exit( EXIT_FAILURE ); \
	}\
}



#define MAX_THREADS_PER_BLOCK 256

inline int divUp(int a, int b) {
	if (b == 0) {
		return 1;
	}
	int result = (a % b != 0) ? (a / b + 1) : (a / b);
	return result;
}


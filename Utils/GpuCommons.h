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



#define MAX_THREADS_PER_BLOCK 1024

inline int divUp(int a, int b) {
	if (b == 0) {
		return 1;
	}
	int result = (a % b != 0) ? (a / b + 1) : (a / b);
	return result;
}



template<typename T>
struct ManagedArray{
    T* data = nullptr;
    int N;

    /*
    in cuda, when an object is passed from a __host__ func to a __global__ func,
    it is copied in the host side first, and the copy is then memcpy'ed on to GPU (constant memeory)
    then, before the kernel starts, the host copy is destroyed.
    Sometimes, we wish to avoid the destructor being called during that destruction,
    because a memcpy'ed object from it is still alive in the GPU.
    */
    bool isCopyForKernel;

    __host__
    ManagedArray(int N_,bool isCopyForKernel_ = false) :N(N_),isCopyForKernel(isCopyForKernel_) {
        if (!isCopyForKernel) {
            std::cout << "allocing Ts " << N << "    " << N * sizeof(T) << std::endl;
            HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
        }
    }

    __host__
    ManagedArray(const ManagedArray& other) {
        isCopyForKernel = other.isCopyForKernel;
        N = other.N;
        if (!isCopyForKernel) {
            std::cout << "allocing Ts bc copy construtor " << N << "    " << N * sizeof(T) << std::endl;
            HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
            HANDLE_ERROR(cudaMemcpy(data, other.data, N * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        else {
            data = other.data;
        }
    }


    ManagedArray(const std::vector<T>& vec) {
        isCopyForKernel = false;
        N = vec.size();
        HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
        HANDLE_ERROR(cudaMemcpy(data, vec.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    }

    __host__
    ManagedArray& operator=(const ManagedArray& other) {
        isCopyForKernel = other.isCopyForKernel;
        N = other.N;
        if (!isCopyForKernel) {
            if (data != nullptr) {
                std::cout << "freeing bc copy assign " << data << std::endl;
                HANDLE_ERROR(cudaFree(data));
            }
            N = other.N;
            std::cout << "allocing Ts bc copy assign" << N << "    " << N * sizeof(T) << std::endl;
            HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
            HANDLE_ERROR(cudaMemcpy(data, other.data, N * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        else {
            data = other.data;
        }
        
    }

    __host__
    ~ManagedArray() {
        if (isCopyForKernel) return;

        CHECK_CUDA_ERROR("before free");
        std::cout << "freeing " << (void*) data << std::endl;

        HANDLE_ERROR(cudaFree(data));
    }

    __host__
    ManagedArray getCopyForKernel() {
        ManagedArray copy(N, true);
        copy.data = data;
        return copy;
    }
};
#pragma once

#include "GpuCommons.h"


template<typename T>
struct CpuArray{
    T* data = nullptr;
    int N;

    /*
    see comments for this varibale in the GpuArray struct
    */
    bool isCopyForKernel;

    __host__
    CpuArray(int N_,bool isCopyForKernel_ = false) :N(N_),isCopyForKernel(isCopyForKernel_) {
        if (!isCopyForKernel) {
            std::cout << "allocing Ts on CPU" << N << "    " << N * sizeof(T) << std::endl;
            data = new T[N];
        }
    }

    __host__
    CpuArray(const CpuArray& other) {
        isCopyForKernel = other.isCopyForKernel;
        N = other.N;
        if (!isCopyForKernel) {
            std::cout << "allocing Ts (on CPU) bc copy construtor " << N << "    " << N * sizeof(T) << std::endl;
            data = new T[N];
            memcpy(data, other.data, N*sizeof(T));
        }
        else {
            data = other.data;
        }
    }


    CpuArray(const std::vector<T>& vec) {
        isCopyForKernel = false;
        N = vec.size();
        data = new T[N];
        memcpy(data, vec.data(), N*sizeof(T));
    }

    __host__
    CpuArray& operator=(const CpuArray& other) {
        isCopyForKernel = other.isCopyForKernel;
        N = other.N;
        if (!isCopyForKernel) {
            if (data != nullptr) {
                std::cout << "freeing (on CPU) bc copy assign " << data << std::endl;
                delete[] data;
            }
            N = other.N;
            std::cout << "allocing Ts bc copy assign" << N << "    " << N * sizeof(T) << std::endl;
            data = new T[N];
            memcpy(data, other.data, N*sizeof(T));
        }
        else {
            data = other.data;
        }
        return *this;
    }

    __host__
    ~CpuArray() {
        if (isCopyForKernel) return;

        std::cout << "freeing " << (void*) data << std::endl;

        delete[] data;
    }

    __host__
    CpuArray getCopyForKernel() {
        CpuArray copy(N, true);
        copy.data = data;
        return copy;
    }
};






template<typename T>
struct GpuArray{
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
    GpuArray(int N_,bool isCopyForKernel_ = false) :N(N_),isCopyForKernel(isCopyForKernel_) {
        if (!isCopyForKernel) {
            std::cout << "allocing Ts " << N << "    " << N * sizeof(T) << std::endl;
            HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
        }
    }

    __host__
    GpuArray(const GpuArray& other) {
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


    GpuArray(const std::vector<T>& vec) {
        isCopyForKernel = false;
        N = vec.size();
        HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
        HANDLE_ERROR(cudaMemcpy(data, vec.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    }

    GpuArray(const CpuArray<T>& vec) {
        isCopyForKernel = false;
        N = vec.N;
        HANDLE_ERROR(cudaMalloc(&data, N * sizeof(T)));
        HANDLE_ERROR(cudaMemcpy(data, vec.data, N * sizeof(T), cudaMemcpyHostToDevice));
    }

    __host__
    GpuArray& operator=(const GpuArray& other) {
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
        return *this;
    }

    __host__
    ~GpuArray() {
        if (isCopyForKernel) return;

        CHECK_CUDA_ERROR("before free");
        std::cout << "freeing " << (void*) data << std::endl;

        HANDLE_ERROR(cudaFree(data));
    }

    __host__
    GpuArray getCopyForKernel() {
        GpuArray copy(N, true);
        copy.data = data;
        return copy;
    }
};

template <typename T>
struct ArrayPair{

    int N;
    bool isCopyForKernel;

    CpuArray<T> cpu;
    GpuArray<T> gpu;

    ArrayPair(int N_,bool isCopyForKernel_ = false): N(N_),isCopyForKernel(isCopyForKernel_),cpu(N_,isCopyForKernel_),gpu(N_,isCopyForKernel_) {
        
    }
};
#pragma once

#include <ctime>
#include "GpuCommons.h"
#include "Array.h"

#include <iostream>

inline long long getSeed(){
    std::time_t result = std::time(nullptr);
    return result;
}


__global__ void initCurandStates ( curandState * states, unsigned long seed, int maxThreads );

struct CurandStateArray:public GpuArray<curandState> {

    __device__ 
    curandState* getState(int index) {
        return data + (index % N);
    }

    __host__
    CurandStateArray(int N_, bool isCopyForKernel_ = false) :GpuArray<curandState>(N_,isCopyForKernel_) {
        if (!isCopyForKernel_) {
            int numThreads = min(N, MAX_THREADS_PER_BLOCK);
            int numBlocks = divUp(N, numThreads);
            initCurandStates << <numBlocks, numThreads >> > (data, getSeed(), N);
            CHECK_CUDA_ERROR("init curand states");
        }
    }

    CurandStateArray getCopyForKernel() {
        CurandStateArray copy(N, true);
        copy.data = data;
        return copy;
    }

};
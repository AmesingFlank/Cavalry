#pragma once

#include <ctime>
#include "GpuCommons.h"
#include "Array.h"

#include <iostream>
#include <vector>

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
/*

__global__ void initSobolCurandStates ( curandStateSobol32 * states,int N, unsigned int* directionVectors);

struct SobolCurandStateArray:public GpuArray<curandStateSobol32> {

    __device__ 
    curandStateSobol32* getState(int index) {
        return data + (index % N);
    }

    __host__
    SobolCurandStateArray(int N_, bool isCopyForKernel_ = false) :GpuArray<curandStateSobol32>(N_,isCopyForKernel_) {
        if (!isCopyForKernel_) {
            std::vector<unsigned int> directionVectorsHost(20000*32);
            auto getDirectionVectorsResult = curandGetDirectionVectors32(directionVectorsHost.data(),CURAND_DIRECTION_VECTORS_32_JOEKUO6);
            if(getDirectionVectorsResult != CURAND_STATUS_SUCCESS ){
                SIGNAL_ERROR("get direction vectors failed\n");
            }
            
            GpuArray<unsigned int> directionVectors = directionVectorsHost;

            int numThreads = min(N, MAX_THREADS_PER_BLOCK);
            int numBlocks = divUp(N, numThreads);
            initSobolCurandStates << <numBlocks, numThreads >> > (data,  N, directionVectors.data);
            CHECK_CUDA_ERROR("init sobol curand states");
        }
    }

    SobolCurandStateArray getCopyForKernel() {
        SobolCurandStateArray copy(N, true);
        copy.data = data;
        return copy;
    }

};
*/
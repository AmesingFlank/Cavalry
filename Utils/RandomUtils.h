#pragma once

#include <ctime>
#include "GpuCommons.h"
#include "Array.h"

#include <iostream>
#include <vector>
#include "MathsCommons.h"


__host__ __device__
inline float2 uniformSampleDisk(float2 randomSource) {
    float r = sqrt(randomSource.x);
    float theta = randomSource.y * 2 * M_PI;
    return make_float2(r * cos(theta), r * sin(theta));
}


__host__ __device__
inline float3 cosineSampleHemisphere(float2 randomSource) {
    float2 xy = uniformSampleDisk(randomSource);
    float zSquared = max(0.f, 1.f - xy.x * xy.x - xy.y * xy.y);
    return make_float3(xy.x, xy.y, sqrt(zSquared));
}

__host__ __device__
inline float3 uniformSampleHemisphere(float2 randomSource) {
    float angle0 = randomSource.x * 2 * M_PI;
    float angle1 = randomSource.y * M_PI / 2.f;
    float x = cos(angle0) * cos(angle1);
    float y = sin(angle0) * cos(angle1);
    float z = sin(angle1);
    return make_float3(x, y, z);
}

__host__ __device__
inline float cosineSampleHemispherePdf(const float3& result) {
    return abs(result.z / M_PI);
}

__host__ __device__
inline float uniformSampleHemispherePdf(const float3& result) {
    return 1.f / (2.f* M_PI);
}


__host__ __device__
inline float3 uniformSampleSphere(const float2& randomSource){
    float u = randomSource.x * 2 * M_PI;
    float v = (randomSource.y - 0.5) * M_PI;
    return make_float3(
		cos(v)*cos(u),
		sin(v),
		cos(v)*sin(u)
	);
}

__host__ __device__
inline float uniformSampleSpherePdf(const float3& result){
    return 1.f / (4.f* M_PI);
}


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
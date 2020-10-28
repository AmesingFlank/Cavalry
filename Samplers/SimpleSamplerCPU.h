#pragma once

#include "../Core/Sampler.h"
#include "../Utils/GpuCommons.h"
#include <random>


class SimpleSamplerCPU: public Sampler{
public:
    
    SimpleSamplerCPU getCopyForKernel();

    __host__ __device__
    virtual float rand1() {
#ifdef __CUDA_ARCH__
        SIGNAL_ERROR("NOT Implemented on GPU");
#else
        return rand()/(float)RAND_MAX;
#endif
    };

    __host__ __device__
    virtual float2 rand2() {
#ifdef __CUDA_ARCH__
        SIGNAL_ERROR("NOT Implemented on GPU");
#else
        return make_float2(rand1(), rand1());
#endif
    };

    __host__ __device__
    virtual float4 rand4() {
#ifdef __CUDA_ARCH__
        SIGNAL_ERROR("NOT Implemented on GPU");
#else
        return make_float4(rand1(), rand1(),rand1(),rand1());
#endif
    };
};
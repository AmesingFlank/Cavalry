#pragma once


#include "../Core/Sampler.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Array.h"


#include <random>
#include <iostream>


class SimpleSampler: public Sampler{
public:

    CurandStateArray states;

    __host__
    SimpleSampler(int samplesPerPixel_,bool isCopyForKernel_ = false);

    __host__ 
    SimpleSampler();

    __host__
    SimpleSampler getCopyForKernel();

    __device__
    virtual int randInt(int N, SamplingState& samplingState)override {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        curandState* myState = states.getState(index);
        return curand(myState) % N;

    }

    __device__
	virtual float rand1(SamplingState& samplingState) override {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        curandState* myState = states.getState(index);
        return curand_uniform(myState);

    };

    __device__
	virtual float2 rand2(SamplingState& samplingState) override {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        curandState* myState = states.getState(index);
        return make_float2(curand_uniform(myState), curand_uniform(myState));
    };


    __device__
	virtual float4 rand4(SamplingState& samplingState) override {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        curandState* myState = states.getState(index);
        return make_float4(curand_uniform(myState), curand_uniform(myState),curand_uniform(myState), curand_uniform(myState));

    };

    //virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, Film& film, int bytesNeededPerSample,int maxSamplesPerRound = -1)  override;
    virtual SamplerObject getObjectFromThis() override;

    virtual int bytesNeededPerThread() override {
        return sizeof(curandState);
    }

};
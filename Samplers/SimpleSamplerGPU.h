#pragma once


#include "../Core/Sampler.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Array.h"


#include <random>
#include <iostream>


class SimpleSamplerGPU: public Sampler{
public:
    int maxThreads;

    CurandStateArray states;

    __host__
    SimpleSamplerGPU(int maxThreads_,bool isCopyForKernel_ = false);

    __host__ 
    SimpleSamplerGPU();

    __host__
    SimpleSamplerGPU getCopyForKernel();

    __device__
    virtual int randInt(int N) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= maxThreads) {
            return;
        }
        curandState* myState = states.getState(index);
        return curand(myState) % N;

    }

    __device__
	virtual float rand1() override {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= maxThreads){
            return;
        }
        curandState* myState = states.getState(index);
        return curand_uniform(myState);

    };

    __device__
	virtual float2 rand2() override {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= maxThreads){
            return;
        }
        curandState* myState = states.getState(index);
        return make_float2(curand_uniform(myState), curand_uniform(myState));

    };


    __device__
	virtual float4 rand4() override {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= maxThreads){
            return;
        }
        curandState* myState = states.getState(index);
        return make_float4(curand_uniform(myState), curand_uniform(myState),curand_uniform(myState), curand_uniform(myState));

    };


    virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, FilmObject& film)  override;

};
#pragma once

#include "../Utils/GpuCommons.h"
#include "Ray.h"
#include "CameraSample.h"
class Camera{
public:
    __host__ __device__
    virtual Ray genRay(const CameraSample& cameraSample) const = 0;
    
    __host__ __device__
    virtual void pdf(const Ray &ray, float* outputPositionProbability, float* outputDirectionProbability) const = 0;

    __device__
    virtual float3 getPosition() const = 0;

    __device__
    virtual float3 getFront() const = 0;
}; 
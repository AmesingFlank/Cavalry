#pragma once

#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"


class BSDF {
public:
    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const = 0;

    __device__
    virtual Spectrum sample(float2 randomSource,float3& incidentOutput, const float3& exitant,float* probabilityOutput) const = 0;

    __device__
    virtual float pdf(const float3& incident, const float3& exitant) const = 0;

    __device__
    virtual bool isDelta() const { return false; };


    // for RL path tracing, we would like to sample from BSDF (instead of weightedQ) if the BSDF is almost delta
    __device__
    virtual bool isAlmostDelta() const { return false; };
};




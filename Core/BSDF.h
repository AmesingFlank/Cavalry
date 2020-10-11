#pragma once

#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"


class BSDF {
public:
    __host__ __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const = 0;
};




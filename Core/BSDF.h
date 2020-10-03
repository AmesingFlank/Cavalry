#pragma once

#include "Color.h"
#include "../Utils/GpuCommons.h"

class BSDF {
public:
    virtual Spectrum eval(float3 incident, float3 exitant) = 0;

};
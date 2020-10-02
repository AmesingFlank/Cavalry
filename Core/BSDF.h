#pragma once

#include "Color.h"
#include "../Utils/GpuCommons.h"

class BSDF {
public:
    virtual Color eval(float3 incident, float3 exitant) = 0;

};
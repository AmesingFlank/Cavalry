#pragma once

#include "../Utils/GpuCommons.h"

struct IntersectionResult{
    bool intersected;
    float distance;
    float3 normal;
    float2 textureCoordinates;
};
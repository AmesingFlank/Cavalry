#pragma once

#include "../Utils/GpuCommons.h"

class Primitive;
struct IntersectionResult{
    bool intersected = false;
    float distance;
    float3 position;
    float3 normal;
    float2 textureCoordinates;
    Primitive* primitive;
};
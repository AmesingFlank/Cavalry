#pragma once

#include "../Utils/GpuCommons.h"
#include "../BSDFs/BSDFObject.h"

class Primitive;

struct IntersectionResult{
    bool intersected = false;
    float distance;
    float3 position;
    float3 normal;
    float2 textureCoordinates;
    const Primitive* primitive;
    BSDFObject bsdf;

    __device__
    void findBSDF();
};
#pragma once

#include "../Utils/GpuCommons.h"

struct Ray{
    float3 origin;
    float3 direction;
    float3 positionAtDistance(float distance) const{
        return origin + distance * direction;
    }
};
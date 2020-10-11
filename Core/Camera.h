#pragma once

#include "../Utils/GpuCommons.h"
#include "Ray.h"
#include "CameraSample.h"
class Camera{
public:
    __host__ __device__
    virtual Ray genRay(const CameraSample& cameraSample) const = 0;
};
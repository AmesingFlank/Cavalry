#pragma once

#include "../Utils/GpuCommons.h"
#include "Ray.h"
#include "CameraSample.h"
class Camera{
public:
    virtual Ray genRay(const CameraSample& cameraSample) const = 0;
};
#pragma once

#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"
#include "CameraSample.h"

class Filter {
public:
    float xwidth = 1.f;
    float ywidth = 1.f;
    __device__
    virtual float contribution(int x, int y, const CameraSample& cameraSample) const = 0;

};


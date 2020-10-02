#pragma once

#include "CameraSample.h"
class Camera{
public:
    Ray genRay(const CameraSample& cameraSample);
};
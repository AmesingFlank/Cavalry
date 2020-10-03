#pragma once

#include "../Utils/GpuCommons.h"

class Primitive;


class VisibilityTest{
public:
    Ray ray;
    float distanceLimit;
    bool useDistanceLimit = false;
    Primitive* sourcePrim = nullptr;
    Primitive* targetPrim = nullptr;
    

    void setDistanceLimit(float limit){
        distanceLimit  = limit;
        useDistanceLimit = true;
    }
};
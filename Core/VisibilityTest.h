#pragma once

#include "../Utils/GpuCommons.h"

class Shape;


class VisibilityTest{
public:
    Ray ray;
    float distanceLimit;
    bool useDistanceLimit = false;
    Shape* sourceGeometry = nullptr;
    Shape* targetGeometry = nullptr;
    

    void setDistanceLimit(float limit){
        distanceLimit  = limit;
        useDistanceLimit = true;
    }
};
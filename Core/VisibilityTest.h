#pragma once

#include "../Utils/GpuCommons.h"


class TriangleMesh;

class VisibilityTest{
public:
    Ray ray;
    float distanceLimit;
    bool useDistanceLimit = false;
    const TriangleMesh* sourceGeometry = nullptr;
    const TriangleMesh* targetGeometry = nullptr;
    
    __host__ __device__
    void setDistanceLimit(float limit){
        distanceLimit  = limit;
        useDistanceLimit = true;
    }
};
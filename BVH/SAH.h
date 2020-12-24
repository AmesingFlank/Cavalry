#pragma once
#include "../Utils/GpuCommons.h"

#define PRIM_INTERSECTION_COST 1.0
#define INTERNAL_TRAVERSAL_COST 1.2

// Surface Area Heuristic

__device__
inline float leafCost(float area, int numPrimitives){
    return area * PRIM_INTERSECTION_COST * numPrimitives;
}

__device__
inline float internalCost(float area, float leftChildCost, float rightChildCost){
    return area * INTERNAL_TRAVERSAL_COST + leftChildCost + rightChildCost;
}

__device__
inline float internalCost(float area, float childrenCost){
    return area * INTERNAL_TRAVERSAL_COST + childrenCost;
}
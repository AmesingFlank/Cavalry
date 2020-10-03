#pragma once

#include "Scene.h"

bool Scene::intersect(IntersectionResult& result, const Ray& ray) const{
    bool foundIntersection = false;
    for(auto& prim:primitives){
        IntersectionResult thisResult;
        if(prim.shape->intersect(thisResult,ray)){
            if(!foundIntersection || thisResult.distance<result.distance){
                result = thisResult;
                foundIntersection = true;
            }
        }
    }
    return foundIntersection;
}

bool Scene::testVisibility(const VisibilityTest& test) const{
    Ray ray = test.ray;
    bool foundIntersection = false;
    for(auto& prim:primitives){
        if(&prim == test.sourcePrim || &prim == test.targetPrim){
            continue;
        }
        IntersectionResult thisResult;
        if(prim.shape->intersect(thisResult,ray)){
            if(test.useDistanceLimit && thisResult.distance <test.distanceLimit){
                return false;
            }
        }
    }
    return true;
}
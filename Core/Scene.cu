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
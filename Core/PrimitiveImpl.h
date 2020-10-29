#pragma once

#include "Primitive.h"

inline void Primitive::buildCpuReferences(const SceneHandle& scene){
    if(areaLightIndex>=0){
        areaLight = &(scene.lights[areaLightIndex]);
    }
    else{
        areaLight = nullptr;
    }
}

inline void Primitive::buildGpuReferences(const SceneHandle& scene){
    if(areaLightIndex>=0){
        areaLight = &(scene.lights[areaLightIndex]);
    }
    else{
        areaLight = nullptr;
    }
}
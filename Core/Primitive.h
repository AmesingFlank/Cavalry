#pragma once

#include "BSDF.h"
#include "../Materials/MaterialObject.h"
#include <vector>
#include <memory>
#include "TriangleMesh.h"

class Scene;
struct SceneHandle;
struct LightObject;

class Primitive{
public:

    MaterialObject material;
    TriangleMesh shape;

    void prepareForRender(Scene& scene,int primID) {
        shape.prepareForRender(scene,primID);
        material.prepareForRender();
    }

    LightObject* areaLight = nullptr;
    int areaLightIndex = -1;
    void setAreaLightIndex(int index) {
        areaLightIndex = index;
    }

    void buildCpuReferences(const SceneHandle& scene);

    __device__
    void buildGpuReferences(const SceneHandle& scene) ;
};
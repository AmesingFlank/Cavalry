#pragma once

#include "BSDF.h"
#include "../Materials/MaterialObject.h"
#include <vector>
#include <memory>
#include "Shape.h"
#include "../Shapes/ShapeObject.h"

struct SceneHandle;
struct LightObject;

class Primitive{
public:

    MaterialObject material;
    ShapeObject shape;

    __host__ __device__
    bool intersect(IntersectionResult& result, const Ray& ray) const{
        if(!shape.intersect(result, ray)) return false;
        result.primitive = this;
        return true;
    }

    void prepareForRender() {
        shape.prepareForRender();
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
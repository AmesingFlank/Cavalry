#pragma once

#include "BSDF.h"
#include "../Materials/MaterialObject.h"
#include <vector>
#include <memory>
#include "Shape.h"
#include "../Shapes/ShapeObject.h"

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
};
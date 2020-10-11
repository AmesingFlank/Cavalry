#pragma once

#include "BSDF.h"
#include "Material.h"
#include <vector>
#include <memory>
#include "Shape.h"
#include "../Shapes/ShapeObject.h"

class Primitive{
public:
    Material material;
    ShapeObject shape;

    __host__ __device__
    bool intersect(IntersectionResult& result, const Ray& ray) const{
        if(!shape.intersect(result, ray)) return false;
        result.primitive = this;
        return true;
    }
};
#pragma once

#include "BSDF.h"
#include "Material.h"
#include <vector>
#include <memory>
#include "IntersectionResult.h"
#include "Ray.h"

class Shape;
// using the this pointer as a unique identifier of the shape.
using ShapeID = const Shape*;


class Shape{
public:
    
    __host__ __device__
    ShapeID getID() const { 
        return this; 
    };

    __host__ __device__
    virtual bool intersect(IntersectionResult& result, const Ray& ray) const = 0;
};
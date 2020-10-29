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

struct SceneHandle;

class Shape{
public:
    
    __host__ __device__
    ShapeID getID() const { 
        return this; 
    };

    __host__ __device__
    virtual bool intersect(IntersectionResult& result, const Ray& ray) const = 0;

    __host__ __device__
    virtual bool area() const = 0;

    __host__ __device__
    virtual IntersectionResult sample(const float4& randomSource, float* outputProbability) const = 0;

    virtual void prepareForRender() {};

    virtual void buildCpuReferences(const SceneHandle& scene)  {};

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene)  {};
};
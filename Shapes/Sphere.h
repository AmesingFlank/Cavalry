
#pragma once

#include "../Core/Shape.h"

class Sphere: public Shape{
public:
    Sphere(float3 center_,float radius_);
    virtual bool intersect(IntersectionResult& result, const Ray& ray) override;

    float3 center;
    float radius;
};
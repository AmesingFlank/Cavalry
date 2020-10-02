#pragma once

#include "BSDF.h"
#include "Material.h"
#include <vector>
#include <memory>
#include "IntersectionResult.h"
#include "Ray.h"


class Shape{
public:
    virtual bool intersect(IntersectionResult& result, const Ray& ray) = 0;
};
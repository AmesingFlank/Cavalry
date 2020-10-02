#pragma once

#include <vector>
#include "Primitive.h"

class Scene{

public:
    std::vector<Primitive> primitives;
    bool intersect(IntersectionResult& result, const Ray& ray) const;
};
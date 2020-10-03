#pragma once

#include <vector>
#include "Primitive.h"
#include "Light.h"

class Scene{

public:
    std::vector<Primitive> primitives;
    std::shared_ptr<EnvironmentMap> environmentMap;
    bool intersect(IntersectionResult& result, const Ray& ray) const;
};
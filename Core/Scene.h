#pragma once

#include <vector>
#include "Primitive.h"
#include "Light.h"
#include "VisibilityTest.h"

class Scene{

public:
    std::vector<Primitive> primitives;
    std::shared_ptr<EnvironmentMap> environmentMap;
    std::vector<std::shared_ptr<Light>> lights;
    bool intersect(IntersectionResult& result, const Ray& ray) const;
    bool testVisibility(const VisibilityTest& test) const;
};
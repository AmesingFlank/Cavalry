#pragma once

#include "BSDF.h"
#include "Material.h"
#include <vector>
#include <memory>
#include "Shape.h"

class Primitive{
public:
    std::unique_ptr<Material> material;
    std::unique_ptr<Shape> shape;
    bool intersect(IntersectionResult& result, const Ray& ray) const;
};
#pragma once

#include "BSDF.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"

class Material{
public:
    std::vector<std::shared_ptr<BSDF>> bsdfs;
    virtual Spectrum eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection);
};
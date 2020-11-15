#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"

class Material{
public:

    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const = 0;

    __device__
    virtual Spectrum eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection) const {

        float cosine = dot(incidentRay.direction, intersection.normal);
        Spectrum result = incidentSpectrum * intersection.bsdf.eval(incidentRay.direction, exitantRay.direction) * cosine;
        return result;
    }

    virtual void prepareForRender() {};
};
#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"

enum class MaterialType: unsigned char {
    Matte = 0, Mirror = 1, Substrate = 2, Metal = 3, Glass = 4
};

class Material{
public:

    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const = 0;

    __device__
    virtual Spectrum eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection) const {

        float cosine = abs(dot(incidentRay.direction, intersection.normal));
        Spectrum result = incidentSpectrum * intersection.bsdf.eval(intersection.worldToLocal(incidentRay.direction), intersection.worldToLocal(exitantRay.direction)) * cosine;
        return result;
    }

    virtual void prepareForRender() {};

    __device__
    virtual MaterialType getType() const  = 0;
};
#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"

class Material{
public:
    BSDFObject bsdf;

    __host__ __device__
    Spectrum eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection) const{
        float cosine = dot(incidentRay.direction,intersection.normal);
        Spectrum result = incidentSpectrum * bsdf.eval(incidentRay.direction,exitantRay.direction) * cosine ;

            
        return result;
    }
};
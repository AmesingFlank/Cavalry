#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"
#include "../Core/Material.h"


class MatteMaterial:public Material{
public:
    LambertianBSDF lambertian;

    MatteMaterial():lambertian(make_float3(100,0,0)){}

    MatteMaterial(float3 color):lambertian(color){}

    __device__
    virtual Spectrum eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection) const override{
        
        
        float cosine = dot(incidentRay.direction,intersection.normal);
        Spectrum result = incidentSpectrum * lambertian.LambertianBSDF::eval(incidentRay.direction,exitantRay.direction) * cosine ;
        return result;
    }

    __device__
    virtual BSDFObject getBSDF() const override {
        return lambertian;
    }
};
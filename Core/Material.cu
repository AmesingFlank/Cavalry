#include "Material.h"

Spectrum Material::eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection){
    
    Spectrum result = make_float3(0,0,0);
    float cosine = dot(incidentRay.direction,intersection.normal);
    for(const auto& bsdf:bsdfs){
        result += bsdf->eval(incidentRay.direction,exitantRay.direction) * incidentSpectrum * cosine ;
    }
    return result;
}
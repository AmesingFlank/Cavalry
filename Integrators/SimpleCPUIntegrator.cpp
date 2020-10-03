#include "SimpleCPUIntegrator.h"

Spectrum SimpleCPUIntegrator::renderRay(const Scene& scene, const Ray& ray){

    IntersectionResult result;
    scene.intersect(result,ray);

    if(result.intersected){
        return make_float3(1,1,1);
    }
    else{
        return make_float3(0,0,0);
    }

}
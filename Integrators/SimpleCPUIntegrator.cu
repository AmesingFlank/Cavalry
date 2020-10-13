#include "SimpleCPUIntegrator.h"

Spectrum SimpleCPUIntegrator::renderRay(const SceneHandle& scene, const Ray& ray, SamplerObject& sampler){

    IntersectionResult result;
    scene.intersect(result,ray);

    if(result.intersected){
        return make_float3(1,1,1);
    }
    else{
        return make_float3(0,0,0);
    }

}
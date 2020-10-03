#include "SimpleCPUIntegrator.h"

Spectrum SimpleCPUIntegrator::renderCameraSample(const Scene& scene, const CameraSample sample){
    Ray ray = camera.genRay(sample);

    IntersectionResult result;
    scene.intersect(result,ray);

    if(result.intersected){
        return make_float3(1,1,1);
    }
    else{
        return make_float3(0,0,0);
    }

}
#include "DirectLightingCPUIntegrator.h"

Spectrum DirectLightingCPUIntegrator::renderRay(const Scene& scene, const Ray& ray){

    IntersectionResult intersection;
    scene.intersect(intersection,ray);

    if(!intersection.intersected){
        return scene.environmentMap->evaluateRay(ray);
    }
    
    Spectrum result = make_float3(0,0,0);

	Primitive* prim = intersection.primitive;

    Ray exitantRay = {intersection.position,ray.direction*-1};

    for(const auto& light: scene.lights){
        Ray rayToLight;
        float probability;
        float2 randomSource = sampler->rand2();

        VisibilityTest visibilityTest;
        visibilityTest.sourceGeometry = prim->shape.get();

        Spectrum incident = light->sampleRayToPoint(intersection.position, randomSource, probability, rayToLight,visibilityTest);
        if(scene.testVisibility(visibilityTest)){
            result += prim->material->eval(rayToLight,incident,exitantRay,intersection);
        }
    }
    return result;
}
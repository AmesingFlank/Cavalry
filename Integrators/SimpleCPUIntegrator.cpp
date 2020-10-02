#include "SimpleCPUIntegrator.h"

RenderResult SimpleCPUIntegrator::render(const Scene& scene, const Camera& camera, Film& film, CameraSampler& cameraSampler) {
    std::vector<CameraSample> allSamples = cameraSampler.genAllSamples(camera, film);


    for(const auto& sample:allSamples){
        Ray ray = camera.genRay(sample);

        IntersectionResult thisResult;
        scene.intersect(thisResult,ray);

		Color color = renderIntersection(thisResult);

        film.addSample(sample,color);
    }
    
    return film.readCurrentResult();
}


Color SimpleCPUIntegrator::renderIntersection(const IntersectionResult& result){
    if(!result.intersected){
        return make_float3(0,0,0);
    }
    else{
        return make_float3(1,1,1);
    }
}
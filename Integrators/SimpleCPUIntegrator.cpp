#include "SimpleCPUIntegrator.h"

RenderResult SimpleCPUIntegrator::render(const Scene& scene, const Camera& camera, Film& film, const CameraSampler& cameraSampler) {
    std::vector<CameraSample> allSamples = cameraSampler.genAllSamples(camera, film);
    std::vector<Ray> allRays;
    std::vector<IntersectionResult> intersectionResults;
    std::vector<Color> allColors;
    for(const auto& sample:allSamples){
        Ray ray = camera.genRay(sample);
        allRays.push_back(ray);

        IntersectionResult thisResult;
        scene->intersect(thisResult,ray);

        intersectionResults.push_back(thisResult);
        allColors.push_back(renderIntersection(result))
    }

    RenderResult renderResult(film.width,film.height);
    for(int i = 0;i<allColors.size();++i){
        writeColorAt(allColors[i], renderResult.data + i*3);
    }
    return renderResult;
}


Color SimpleCPUIntegrator::renderIntersection(const IntersectionResult& result){
    if(result.intersected){
        return make_float3(0,0,0);
    }
    else{
        return make_float3(1,1,1);
    }
}
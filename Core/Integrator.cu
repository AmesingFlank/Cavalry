#include "Integrator.h"

RenderResult SamplingIntegrator::render(const Scene& scene, const Camera& camera, Film& film){
    std::vector<CameraSample> allSamples = cameraSampler->genAllSamples(camera, film);


    for(const auto& sample:allSamples){
        Spectrum color = renderCameraSample(scene,sample);
        film.addSample(sample,color);
    }
    
    return film.readCurrentResult();
}
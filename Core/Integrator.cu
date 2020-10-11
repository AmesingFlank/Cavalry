#include "Integrator.h"

RenderResult SamplingIntegratorCPU::render(const Scene& scene, const CameraObject& camera, Film& film){
    std::vector<CameraSample> allSamples = cameraSampler->genAllSamples(camera, film);

    SceneHandle sceneHandle= scene.getHostHandle();

    for(const auto& sample:allSamples){
        Spectrum color = renderRay(sceneHandle,camera.genRay(sample));
        film.addSample(sample,color);
    }
    
    return film.readCurrentResult();
}

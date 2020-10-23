#include "DirectLightingGPUIntegrator.h"
#include "../Samplers/SimpleSamplerGPU.h"

DirectLightingGPUIntegrator::DirectLightingGPUIntegrator() {
    
}

RenderResult DirectLightingGPUIntegrator::render(const Scene& scene, const CameraObject& camera, FilmObject& film) {
    sampler = std::make_unique<SamplerObject>( SimpleSamplerGPU( film.getWidth()*film.getHeight()) );
    return SamplingIntegratorGPU<DirectLightingGPUIntegrator>::render(scene, camera, film); 
}
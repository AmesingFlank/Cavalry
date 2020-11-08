#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

class DirectLightingGPUIntegrator : public Integrator{
public:


	DirectLightingGPUIntegrator();

    virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) override;

    
    
};
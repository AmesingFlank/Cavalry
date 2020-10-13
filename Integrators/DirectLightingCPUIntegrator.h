#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

class DirectLightingCPUIntegrator : public SamplingIntegratorCPU<DirectLightingCPUIntegrator> {
public:
	DirectLightingCPUIntegrator();
	static Spectrum renderRay(const SceneHandle& scene, const Ray& ray,SamplerObject& sampler);
    
};
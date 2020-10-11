#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

class DirectLightingCPUIntegrator : public SamplingIntegratorCPU {
public:
	virtual Spectrum renderRay(const SceneHandle& scene, const Ray& ray) override;

    
};
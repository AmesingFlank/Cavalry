#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

class DirectLightingCPUIntegrator : public SamplingIntegrator {
public:
	virtual Spectrum renderRay(const Scene& scene, const Ray& ray) override;

    
};
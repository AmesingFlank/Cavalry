#pragma once

#include "../Core/Integrator.h"

class SimpleCPUIntegrator : public SamplingIntegratorCPU<SimpleCPUIntegrator> {
public:
	static Spectrum renderRay(const SceneHandle& scene, const Ray& ray,SamplerObject& sampler);
};
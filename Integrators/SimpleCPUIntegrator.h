#pragma once

#include "../Core/Integrator.h"

class SimpleCPUIntegrator : public SamplingIntegratorCPU {
public:
	virtual Spectrum renderRay(const SceneHandle& scene, const Ray& ray) override;
};
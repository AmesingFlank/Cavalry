#pragma once

#include "../Core/Integrator.h"

class SimpleCPUIntegrator : public SamplingIntegrator {
public:
	virtual Spectrum renderRay(const Scene& scene, const Ray& ray) override;
};
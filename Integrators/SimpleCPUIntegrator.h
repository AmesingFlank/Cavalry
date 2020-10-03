#pragma once

#include "../Core/Integrator.h"

class SimpleCPUIntegrator : public SamplingIntegrator {
public:
	virtual Spectrum renderCameraSample(const Scene& scene, const CameraSample sample) override;
};
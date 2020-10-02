#pragma once

#include "../Core/Integrator.h"

class SimpleCPUIntegrator : public Integrator {
public:
	virtual RenderResult render(const Scene& scene, const Camera& camera, Film& film, CameraSampler& cameraSampler) override;

	Color renderIntersection(const IntersectionResult& result);
};
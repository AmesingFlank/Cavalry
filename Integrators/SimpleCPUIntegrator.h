#pragma once

#include "../Core/Integrator.h"

class SimpleCPUIntegrator: public Integrator {
public:
    RenderResult render(const Scene& scene, const Camera& camera, Film& film,const CameraSampler& cameraSampler) override;

    Color renderIntersection(const IntersectionResult& result);
}
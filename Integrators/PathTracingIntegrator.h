#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace PathTracing {
    class PathTracingIntegrator : public Integrator {
    public:

        int maxDepth;
        PathTracingIntegrator(int maxDepth_);

        virtual void render(const Scene& scene, const CameraObject& camera, FilmObject& film) override;

    };
}


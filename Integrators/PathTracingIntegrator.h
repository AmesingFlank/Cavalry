#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace PathTracing {
    class PathTracingIntegrator : public Integrator {
    public:


        PathTracingIntegrator();

        virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) override;

    };
}


#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace DirectLighting {
    class DirectLightingIntegrator : public Integrator {
    public:


        DirectLightingIntegrator();

        virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) override;

    };
}


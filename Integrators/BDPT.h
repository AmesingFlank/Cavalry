#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace BDPT {
    class BDPTIntegrator : public Integrator {
    public:

        int maxDepth;
        BDPTIntegrator(int maxDepth_);

        virtual void render(const Scene& scene, const CameraObject& camera, FilmObject& film) override;

    };
}


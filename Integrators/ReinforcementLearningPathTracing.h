#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace ReinforcementLearningPathTracing {


    class RLPTIntegrator : public Integrator {
    public:

        int maxDepth;
        RLPTIntegrator(int maxDepth_);

        virtual void render(const Scene& scene, const CameraObject& camera, Film& film) override;

    };
}


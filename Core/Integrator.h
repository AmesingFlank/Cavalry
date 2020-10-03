#pragma once

#include "Scene.h"
#include "Film.h"
#include "Camera.h"
#include "Sampler.h"
#include "Color.h"
#include <memory>


class Integrator{
public:
    virtual RenderResult render(const Scene& scene, const Camera& camera, Film& film) = 0;
};


class SamplingIntegrator{
public:
    std::unique_ptr<CameraSampler> cameraSampler;
    virtual RenderResult render(const Scene& scene, const Camera& camera, Film& film) override;
    virtual Spectrum renderCameraSample(const Scene& scene, const CameraSample sample) = 0;
};
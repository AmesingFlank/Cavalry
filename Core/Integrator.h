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
    std::unique_ptr<Sampler> sampler;
};


class SamplingIntegrator: public Integrator{
public:
    std::unique_ptr<CameraSampler> cameraSampler;
    virtual RenderResult render(const Scene& scene, const Camera& camera, Film& film) override;
    virtual Spectrum renderRay(const Scene& scene, const Ray& ray) = 0;
};
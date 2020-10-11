#pragma once

#include "Scene.h"
#include "Film.h"
#include "Camera.h"
#include "Sampler.h"
#include "Color.h"
#include <memory>
#include "../Cameras/CameraObject.h"


class Integrator{
public:
    virtual RenderResult render(const Scene& scene, const CameraObject& camera, Film& film) = 0;
    std::unique_ptr<Sampler> sampler;
};


class SamplingIntegratorCPU: public Integrator{
public:
    std::unique_ptr<CameraSampler> cameraSampler;
    virtual RenderResult render(const Scene& scene, const CameraObject& camera, Film& film) override;

    virtual Spectrum renderRay(const SceneHandle& scene, const Ray& ray) = 0;
};


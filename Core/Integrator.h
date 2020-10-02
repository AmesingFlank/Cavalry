#pragma once

#include "Scene.h"
#include "Film.h"
#include "Camera.h"
#include "Sampler.h"
class Integrator{
public:
    virtual RenderResult render(const Scene& scene, const Camera& camera, Film& film, CameraSampler& cameraSampler) = 0;
};
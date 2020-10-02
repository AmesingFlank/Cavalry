#pragma once

#include "Scene.h"
#include "Film.h"
#include "Camera.h"

class Integrator{
public:
    RenderResult render(const Scene& scene, const Camera& camera, Film& film,const CameraSampler& cameraSample);
};
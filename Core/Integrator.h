#pragma once

#include "Scene.h"
#include "Film.h"
#include "Camera.h"
#include "Sampler.h"
#include "Color.h"
#include <memory>
#include "../Cameras/CameraObject.h"
#include "../Samplers/SamplerObject.h"

class Integrator{
public:
    virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) = 0;
    std::unique_ptr<SamplerObject> sampler;
};





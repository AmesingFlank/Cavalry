#pragma once

#include "RenderResult.h"
#include "Integrator.h"
#include "Camera.h"
#include "Film.h"
#include "Sampler.h"
#include <memory>
#include "../Cameras/CameraObject.h"
#include "../Films/FilmObject.h"


class Renderer {
public:
    std::unique_ptr<Integrator> integrator;
    std::unique_ptr<CameraObject> camera;
    std::unique_ptr<FilmObject> film;

	RenderResult render(const Scene& scene);
};
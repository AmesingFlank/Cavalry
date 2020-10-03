#pragma once

#include "RenderResult.h"
#include "Integrator.h"
#include "Camera.h"
#include "Film.h"
#include "Sampler.h"
#include <memory>

class Renderer {
public:
    std::unique_ptr<Integrator> integrator;
    std::unique_ptr<Camera> camera;
    std::unique_ptr<Film> film;

	RenderResult render(const Scene& scene);
};
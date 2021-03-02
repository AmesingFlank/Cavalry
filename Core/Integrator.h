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
    virtual void render(const Scene& scene, const CameraObject& camera, Film& film) = 0;
    std::unique_ptr<SamplerObject> sampler;

    // a single call to render might not finish the entire rendering task.
    virtual bool isFinished(const Scene& scene, const CameraObject& camera, Film& film) {
        return film.completedSamplesPerPixel == sampler->getSamplesPerPixel();
    }    
};





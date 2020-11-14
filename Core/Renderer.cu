#include "Renderer.h"

RenderResult Renderer::render(const Scene& scene){
    integrator->render(scene,*camera,*film);
    return film->readCurrentResult();
}
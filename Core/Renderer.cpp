#include "Renderer.h"

RenderResult Renderer::render(const Scene& scene){
    return integrator->render(scene,*camera,*film);
}
#include <iostream>

#include "Core/Renderer.h"
#include "Core/Material.h"
#include "Shapes/Sphere.h"
#include "Integrators/SimpleCPUIntegrator.h"
#include "Cameras/PerspectiveCamera.h"
#include "Samplers/NaiveCameraSampler.h"
#include "Films/SimpleFilm.h"
#include "BSDFs/Lambertian.h"

int main(){

    

    Renderer renderer;

    renderer->integrator = std::make_unique<SimpleCPUIntegrator>();
    renderer->camera = std::make_unique<PerspectiveCamera>();
    renderer->film = std::make_unique<SimpleFilm>(200,200);
    renderer->cameraSampler = std::make_unique<CameraSampler>();


    Material lambertian;
    lambertian.bsdfs.push_back(LambertianBSDF());

    Scene scene;
    
    Primitive prim0;
    prim0->shape = std::make_unique<Sphere>(make_float3(0,0,5),3);
    prim0->material = std::make_unique<Material>(lambertian);

    scene.primitives.push_back(std::move(prim0));

    renderer.render(scene).saveToPNG("test.png");
}
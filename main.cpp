#include <iostream>

#include "Core/Renderer.h"
#include "Core/Material.h"
#include "Shapes/Sphere.h"
#include "Integrators/SimpleCPUIntegrator.h"
#include "Integrators/DirectLightingCPUIntegrator.h"
#include "Cameras/PerspectiveCamera.h"
#include "Samplers/NaiveCameraSampler.h"
#include "Films/SimpleFilm.h"
#include "BSDFs/Lambertian.h"
#include "Samplers/SimpleSampler.h"
#include "Lights/PointLight.h"

int main(){

    

    Renderer renderer;

    std::unique_ptr<DirectLightingCPUIntegrator> integrator = std::make_unique<DirectLightingCPUIntegrator>();
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();
    integrator->sampler = std::make_unique<SimpleSampler>();


    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<PerspectiveCamera>();
    renderer.film = std::make_unique<SimpleFilm>(1000,1000);


    Material lambertian;
    lambertian.bsdfs.push_back(std::make_shared<LambertianBSDF>(make_float3(1,1,1)));

    Scene scene;
    
    Primitive prim0;
    prim0.shape = std::make_unique<Sphere>(make_float3(0,1,7),0.7);
    prim0.material = std::make_unique<Material>(lambertian);
    scene.primitives.push_back(std::move(prim0));

	Primitive prim1;
	prim1.shape = std::make_unique<Sphere>(make_float3(0,-100,17),100);
	prim1.material = std::make_unique<Material>(lambertian);
	scene.primitives.push_back(std::move(prim1));

    std::shared_ptr<EnvironmentMap> environmentMap = std::make_shared<EnvironmentMap>();
    std::shared_ptr<PointLight> light0 = std::make_shared<PointLight>(make_float3(0,7,2),make_float3(1,1,1));

    scene.environmentMap = environmentMap;
    scene.lights.push_back(environmentMap);
    scene.lights.push_back(light0);

    renderer.render(scene).saveToPNG("test.png");
}
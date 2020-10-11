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

#include <variant>



int main(){


    Renderer renderer;

    std::unique_ptr<DirectLightingCPUIntegrator> integrator = std::make_unique<DirectLightingCPUIntegrator>();
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();
    integrator->sampler = std::make_unique<SimpleSampler>();


    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<CameraObject>(PerspectiveCamera());
    renderer.film = std::make_unique<SimpleFilm>(1000,1000);


    Material lambertian;
    lambertian.bsdf = LambertianBSDF(make_float3(1,1,1));

    Scene scene;
    
    Primitive prim0;
    prim0.shape = Sphere(make_float3(0,1,7),0.7);
    prim0.material = Material(lambertian);
    scene.primitivesHost.push_back(prim0);

	Primitive prim1;
	prim1.shape =  Sphere(make_float3(0,-100,17),100);
	prim1.material = Material(lambertian);
	scene.primitivesHost.push_back(prim1);


    scene.lightsHost.push_back(EnvironmentMap());
    scene.lightsHost.push_back(PointLight(make_float3(0,7,2),make_float3(1,1,1)));
    scene.environmentMapIndex = 0;

    renderer.render(scene).saveToPNG("test.png");
}
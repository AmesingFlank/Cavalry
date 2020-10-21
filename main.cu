#include <iostream>

#include "Core/Renderer.h"
#include "Core/Material.h"
#include "Shapes/Sphere.h"
#include "Integrators/SimpleCPUIntegrator.h"
#include "Integrators/DirectLightingCPUIntegrator.h"
#include "Integrators/DirectLightingGPUIntegrator.h"

#include "Cameras/PerspectiveCamera.h"
#include "Samplers/NaiveCameraSampler.h"
#include "BSDFs/Lambertian.h"
#include "Lights/PointLight.h"
#include "Films/FilmObject.h"
#include <variant>


Scene testScene0() {
    Material lambertian;
    lambertian.bsdf = LambertianBSDF(make_float3(1, 1, 1));

    Scene scene;


    Primitive prim0;
    prim0.shape = Sphere(make_float3(0, 1, -7), 0.7);
    prim0.material = Material(lambertian);
    scene.primitivesHost.push_back(prim0);

    Primitive prim1;
    prim1.shape = Sphere(make_float3(0, -100, -17), 100);
    prim1.material = Material(lambertian);
    scene.primitivesHost.push_back(prim1);



    Primitive prim2;
    TriangleMesh mesh(1, 3, false, false);
    mesh.positions.cpu.data[0] = make_float3(0, 0, -3);
    mesh.positions.cpu.data[1] = make_float3(1, 0, -3);
    mesh.positions.cpu.data[2] = make_float3(0, 1, -6);
    mesh.indices.cpu.data[0] = make_int3(1, 0, 2);
    mesh.copyToDevice();

    prim2.shape = mesh;
    prim2.material = Material(lambertian);
    scene.primitivesHost.push_back(prim2);



    scene.lightsHost.push_back(EnvironmentMap());
    scene.lightsHost.push_back(PointLight(make_float3(0, 7, 2), make_float3(1, 1, 1)));
    scene.environmentMapIndex = 0;

    return scene;

}

void testSimpleCPU0() {
    Renderer renderer;


    std::unique_ptr<SimpleCPUIntegrator> integrator = std::make_unique<SimpleCPUIntegrator>();
    
    
    int width = 1000,height = 1000;
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

    PerspectiveCamera camera(make_float3(0,0,0),make_float3(0,0,-1),make_float3(0,1,0),45,width,height);

    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<CameraObject>(camera);
    renderer.film = std::make_unique<FilmObject>(SimpleFilmGPU(width,height));

    Scene scene = testScene0();

    scene.copyToDevice();

    renderer.render(scene).saveToPNG("test.png");
}


void testDirectLightingCPU0() {
    Renderer renderer;

    std::unique_ptr<DirectLightingCPUIntegrator> integrator = std::make_unique<DirectLightingCPUIntegrator>();

    
    int width = 1000,height = 1000;
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

    PerspectiveCamera camera(make_float3(0, 0, 0), make_float3(0, 0, -1), make_float3(0, 1, 0), 45, width, height);

    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<CameraObject>(camera);
    renderer.film = std::make_unique<FilmObject>(SimpleFilmCPU(width,height));



    Scene scene = testScene0();


    scene.copyToDevice();

    renderer.render(scene).saveToPNG("test.png");
}

void testDirectLightingGPU0() {
    Renderer renderer;

    std::unique_ptr<DirectLightingGPUIntegrator> integrator = std::make_unique<DirectLightingGPUIntegrator>();


    int width = 1000,height = 1000;
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

    PerspectiveCamera camera(make_float3(-3,0,0),make_float3(0, 1,-7),make_float3(0,1,0),45,width,height);

    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<CameraObject>(camera);
    renderer.film = std::make_unique<FilmObject>(SimpleFilmGPU(width,height));

    Scene scene = testScene0();

    scene.copyToDevice();

    renderer.render(scene).saveToPNG("test.png");
}


int main(){

    testDirectLightingGPU0();
    
}
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
#include "SceneLoading/SceneLoading.h"
#include "Utils/MathsCommons.h"

Scene testScene1() {
    Material lambertian;
    lambertian.bsdf = LambertianBSDF(make_float3(1, 1, 1));

    Scene scene;


    Primitive prim3;

    TriangleMesh dragon = TriangleMesh::createFromPLY("../TestScenes/head.ply", glm::mat4(1.0));
    dragon.copyToDevice();

    prim3.shape = dragon;
    prim3.material = Material(lambertian);
    scene.primitivesHost.push_back(prim3);



    scene.lightsHost.push_back(EnvironmentMap());
    //scene.lightsHost.push_back(PointLight(make_float3(0, 7, 2), make_float3(1, 1, 1)));
    scene.environmentMapIndex = 0;

    return scene;

}



Scene testScene0() {
    Material lambertian;
    lambertian.bsdf = LambertianBSDF(make_float3(1, 1, 1));

    Scene scene;

    

    Primitive prim0;
    prim0.shape = Sphere(make_float3(0, 1, -3), 0.7);
    prim0.material = Material(lambertian);
    scene.primitivesHost.push_back(prim0);

    Primitive prim1;
    prim1.shape = Sphere(make_float3(0, -100, -17), 100);
    prim1.material = Material(lambertian);
    scene.primitivesHost.push_back(prim1);



    Primitive prim2;
    TriangleMesh mesh(1, 3, false, false);
    mesh.positions.cpu.data[0] = make_float3(0, 0, -1);
    mesh.positions.cpu.data[1] = make_float3(1, 0, -1);
    mesh.positions.cpu.data[2] = make_float3(0, 2, -3);
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
    
    
    int width = 128,height = 128;
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

    glm::mat4 cameraToWorld = glm::inverse(glm::lookAt(glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f,1.f,0.f) ));

    PerspectiveCamera camera(cameraToWorld,45,width,height);

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

    
    int width = 512,height = 512;
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

    glm::mat4 cameraToWorld = glm::inverse(glm::lookAt(glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)));
    PerspectiveCamera camera(cameraToWorld, glm::radians(45.f), width, height);

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

    glm::mat4 cameraToWorld = glm::inverse(glm::lookAt(glm::vec3(-1, 0.5, 1)*3.f, glm::vec3(0, 0, -1), glm::vec3(0, 1, 0)));

    //glm::mat4 cameraToWorld = glm::inverse(glm::lookAt(glm::vec3(0, 0,10), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0)));


    PerspectiveCamera camera(cameraToWorld, glm::radians(45.f), width, height);


    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<CameraObject>(camera);
    renderer.film = std::make_unique<FilmObject>(SimpleFilmGPU(width,height));

    Scene scene = testScene0();

    scene.copyToDevice();

    renderer.render(scene).saveToPNG("test.png");
}

void testDirectLightingGPU1() {
    Renderer renderer;

    std::unique_ptr<DirectLightingGPUIntegrator> integrator = std::make_unique<DirectLightingGPUIntegrator>();


    int width = 1280, height = 720;
    integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

    glm::mat4 cameraToWorld = glm::inverse(glm::lookAt(
        glm::vec3(0.322839, 0.0534825, 0.504299), 
        glm::vec3(-0.140808, -0.162727, -0.354936),
        glm::vec3(0.0355799, 0.964444, -0.261882)));
    PerspectiveCamera camera(cameraToWorld, glm::radians(30.f), width, height);


    renderer.integrator = std::move(integrator);
    renderer.camera = std::make_unique<CameraObject>(camera);
    renderer.film = std::make_unique<FilmObject>(SimpleFilmGPU(width, height));

    Scene scene = testScene1();

    scene.copyToDevice();

    renderer.render(scene).saveToPNG("test.png");
}

void testParsing0(){
    RenderSetup setup = readRenderSetup("../TestScenes/cornellBox/scene.pbrt");
    setup.scene.lightsHost.push_back(PointLight(make_float3(0,2,3), make_float3(1, 1, 1)));
    setup.scene.copyToDevice();
    setup.renderer.render(setup.scene).saveToPNG("test.png");
}


int main(){

    testParsing0();
    //testDirectLightingGPU1();
    //testSimpleCPU0();
}
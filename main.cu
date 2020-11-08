#include <iostream>

#include "Core/Renderer.h"
#include "Core/Material.h"

#include "Integrators/DirectLightingGPUIntegrator.h"

#include "Cameras/PerspectiveCamera.h"

#include "BSDFs/Lambertian.h"
#include "Lights/PointLight.h"
#include "Lights/DiffuseAreaLight.h"

#include "Films/FilmObject.h"
#include <variant>
#include "SceneLoading/SceneLoading.h"
#include "Utils/MathsCommons.h"
#include "Materials/MaterialObject.h"

#include "Utils/Timer.h"



void testParsingCornell() {
    RenderSetup setup = readRenderSetup("../TestScenes/cornellBox/scene.pbrt");
    //setup.scene.lightsHost.push_back(PointLight(make_float3(0, 2, 3), make_float3(1, 1, 1)));
    setup.scene.prepareForRender();
    setup.renderer.render(setup.scene).saveToPNG("test.png");
}

void testParsingBath() {
    RenderSetup setup = readRenderSetup("C:\\Users/Dunfan/Code/pbrt/pbrt-v3-scenes/bathroom/bathroom.pbrt");
    //RenderSetup setup = readRenderSetup("../TestScenes/bathroom/bathroom.pbrt");
    Timer::getInstance().start("preparation");
    Timer::getInstance().start("all");
    setup.scene.prepareForRender();
    Timer::getInstance().stop("preparation");
    setup.renderer.render(setup.scene).saveToPNG("test.png");
    Timer::getInstance().stop("all");
    Timer::getInstance().printResults();
}


int main(){
    testParsingBath();
    //testParsingCornell();
    //testParsingHead();
    //testDirectLightingCPU0();

    std::cout << "done!" << std::endl;
}
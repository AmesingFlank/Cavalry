#include <iostream>

#include "Core/Renderer.h"
#include "Core/Material.h"

#include "Integrators/DirectLightingIntegrator.h"

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





void test(const std::string& scenePath) {
    RenderSetup setup = readRenderSetup(scenePath);

    Timer::getInstance().start("preparation");
    Timer::getInstance().start("all");
    setup.scene.prepareForRender();
    Timer::getInstance().stop("preparation");
    setup.renderer.render(setup.scene).saveToPNG(setup.outputFileName);
    Timer::getInstance().stop("all");
    Timer::getInstance().printStatistics();
}


int main(){
    //test("../TestScenes/cornellBox/test.pbrt");
    //test("../TestScenes/bathroom2/scene.pbrt");
    test("../TestScenes/living-room-3/scene.pbrt");
    //test("../TestScenes/staircase/scene.pbrt");

}
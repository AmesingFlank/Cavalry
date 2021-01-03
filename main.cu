#include <iostream>

#include "Core/Renderer.h"

#include "Films/FilmObject.h"
#include <variant>
#include "SceneLoading/SceneLoading.h"

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

    std::string path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/";

    //test(path+"TestScenes/cornellBox/scene.pbrt");
    //test(path+"TestScenes/bathroom2/scene.pbrt");
    //test(path+"TestScenes/bathroom/bathroom.pbrt");
    //test(path+"TestScenes/living-room-3/scene.pbrt");
    //test(path+"TestScenes/staircase/scene.pbrt");
    //test(path+"TestScenes/staircase2/scene.pbrt");
    //test(path+"TestScenes/ganesha/ganesha.pbrt");
    //test(path+"TestScenes/coffee/scene.pbrt");
    //test(path + "TestScenes/veach-mis2/scene.pbrt");
    test(path + "TestScenes/veach-mis/mis.pbrt");

}
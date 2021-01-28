#include <iostream>

#include "Core/Renderer.h"

#include "Films/FilmObject.h"
#include <variant>
#include "SceneLoading/SceneLoading.h"

#include "Utils/Timer.h"





void test(const std::string& scenePath) {
    RenderSetup setup = readRenderSetup(scenePath);

    Timer::getInstance().start("all");
    Timer::getInstance().start("preparation");
    setup.scene.prepareForRender();
    Timer::getInstance().stop("preparation");
    auto& result = setup.renderer.render(setup.scene);
    Timer::getInstance().stop("all");
    result.saveToPNG(setup.getOutputFileName());
    Timer::getInstance().printStatistics();
}


int main(){

    std::string path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/";

  test(path+"TestScenes/cornellBox/test5.pbrt");
    //test(path+"TestScenes/bathroom2/scene.pbrt");
    //test(path+"TestScenes/bathroom/bathroom.pbrt");
    //test(path+"TestScenes/living-room-3/scene.pbrt");
  //test(path+"TestScenes/staircase/scene.pbrt");
    //test(path+"TestScenes/staircase2/scene.pbrt");
    //test(path+"TestScenes/ganesha/ganesha.pbrt");
    //test(path+"TestScenes/coffee/scene.pbrt");
    //test(path + "TestScenes/veach-mis2/scene.pbrt");
    //test(path + "TestScenes/veach-mis/mis.pbrt");
    //test(path + "TestScenes/veach-bidir2/scene.pbrt");
    //test(path + "TestScenes/dragon/f8-21b.pbrt");
    //test(path+"TestScenes/dragon/f9-3.pbrt");
}
#include <iostream>

#include "Core/Renderer.h"

int main(){
    Renderer renderer;
    renderer.render().saveToPNG("test.png");
}
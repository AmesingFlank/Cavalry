#include "SimpleFilm.cu"

SimpleFilm::SimpleFilm(int width_, int height_):width(width_), height(height_),result(width_,height_){
    
}

RenderResult SimpleFilm::readCurrentResult(){
    return result;
}

void SimpleFilm::addSample(float2 position, Color color){
    int x = position.x;
    int y = position.y;
    int index = y*width + x;
    writeColorAt(color,result.data+index*3);
}
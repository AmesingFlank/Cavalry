#pragma once

#include "../Core/Shape.h"
#include "TriangleMesh.h"


class Triangle:public Shape {
public:

    int meshIndex;
    TriangleMesh* mesh;
    int triangleIndex;

    Trinagle(){}

    Triangle(int triangleIndex_):triangleIndex(triangleIndex_){
        
    }

};
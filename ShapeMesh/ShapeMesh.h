#pragma once

#include "../Core/TriangleMesh.h"
#include "../Core/Parameters.h"
#include "Sphere.h"
namespace ShapeMesh{
    inline TriangleMesh createShapeMesh(const ObjectDefinition& def, const glm::mat4& transform){
        if(def.objectName == "sphere"){
            float radius = def.params.getNum("radius");
            return createSphereMesh(radius,transform);
        }
        SIGNAL_ERROR("ShapeMesh type not supported: %s\n",def.objectName.c_str());
    }
}


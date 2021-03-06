#pragma once
#include "Triangle.h"

// this file is needed so that implementations of these functions needs to be out-of-class,
// which is because the class definitions includes incomplete defitions.
// These function implementations are put inside an *Impl.h file instead of a .cpp/.cu file.
// in order to avoid requiring relocatable GPU code (i.e., -rdc=true).


inline void Triangle::buildCpuReferences(const SceneHandle& scene) {
    Primitive* prim = &scene.primitives[meshIndex];
    mesh = &(prim->shape);
}

__device__
inline void Triangle::buildGpuReferences(const SceneHandle& scene) {
    Primitive* prim = &scene.primitives[meshIndex];
    mesh = &(prim->shape);
}
#pragma once

#include "TriangleMesh.h"

// this file is needed so that implementations of these functions needs to be out-of-class,
// which is because the class definitions includes incomplete defitions.
// These function implementations are put inside an *Impl.h file instead of a .cpp/.cu file.
// in order to avoid requiring relocatable GPU code (i.e., -rdc=true).


inline void TriangleMesh::buildCpuReferences(const SceneHandle& scene,int primIndex) {
    meshIndex = primIndex;
    prim = &scene.primitives[primIndex];
}

__device__
inline void TriangleMesh::buildGpuReferences(const SceneHandle& scene,int primIndex) {
    meshIndex = primIndex;
    prim = &scene.primitives[primIndex];
}
#include "Triangle.h"

inline void Triangle::buildCpuReferences(const SceneHandle& scene) {
    prim = &scene.primitives[meshIndex];
    mesh = &(prim->shape);
}

__device__
inline void Triangle::buildGpuReferences(const SceneHandle& scene) {
    prim = &scene.primitives[meshIndex];
    mesh = &(prim->shape);
}
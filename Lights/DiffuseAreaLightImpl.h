#include "DiffuseAreaLight.h"

// this file is needed so that implementations of these functions needs to be out-of-class,
// which is because the class definitions includes incomplete defitions.
// These function implementations are put inside an *Impl.h file instead of a .cpp/.cu file.
// in order to avoid requiring relocatable GPU code (i.e., -rdc=true).


inline void DiffuseAreaLight::buildCpuReferences(const SceneHandle& scene) {
    shape = &(scene.primitives[shapeIndex].shape);
}

__device__
inline void DiffuseAreaLight::buildGpuReferences(const SceneHandle& scene) {
    
    shape = &(scene.primitives[shapeIndex].shape);
    printf("here in build %p %f %f %f\n", (void*)shape, color.x, color.y, color.z);
    printf("built gpu! %d\n", shapeIndex);
    printf("device size:%d\n", sizeof(LightObject));
}
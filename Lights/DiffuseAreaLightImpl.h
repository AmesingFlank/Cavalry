#include "DiffuseAreaLight.h"

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
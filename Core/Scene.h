#pragma once

#include <vector>
#include "Primitive.h"
#include "Light.h"
#include "VisibilityTest.h"
#include "../Lights/LightObject.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include "../Utils/Array.h"
#include "Triangle.h"
#include "../BVH/BVH.h"


struct SceneHandle{
    Primitive* mutable primitives;
    size_t primitivesCount;

    Triangle* triangles;
    size_t trianglesCount;

    LightObject* mutable lights;
    size_t lightsCount;

    const LightObject* environmentMapLightObject;

    BVH bvh;

    AABB sceneBounds;

    __host__ __device__
    const EnvironmentMap* getEnvironmentMap() const{
        return environmentMapLightObject->get<EnvironmentMap>();
    }

    __host__ __device__
    bool hasEnvironmentMap() const {
        return environmentMapLightObject != nullptr;
    }
    
    
    __device__
    bool intersect(IntersectionResult& result, const Ray& ray) const{

        if (bvh.intersect(result, ray, triangles)) {
            result.findBSDF();
            return true;
        }
        return false;

    }

    __device__
    bool testVisibility(const VisibilityTest& test) const{
        return bvh.testVisibility(test, triangles);
    }
};

class Scene{
public:
    std::vector<Primitive> primitivesHost;
    GpuArray<Primitive> primitivesDevice = GpuArray<Primitive>(0,true);

    std::vector<Triangle> trianglesHost;
    GpuArray<Triangle> trianglesDevice = GpuArray<Triangle>(0,true);

    std::vector<LightObject> lightsHost;
    GpuArray<LightObject> lightsDevice = GpuArray<LightObject>(0,true);

    BVH bvh;

    int environmentMapIndex  = -1;

    AABB sceneBounds;

    SceneHandle getHostHandle() const;

    SceneHandle getDeviceHandle() const;

    void prepareForRender();

    void copyToDevice();

    void buildCpuReferences() ;
    void buildGpuReferences() ;
    
};

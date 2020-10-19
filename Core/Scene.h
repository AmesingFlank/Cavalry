#pragma once

#include <vector>
#include "Primitive.h"
#include "Light.h"
#include "VisibilityTest.h"
#include "../Lights/LightObject.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include "../Utils/Array.h"

struct SceneHandle{
    const Primitive* primitives;
    size_t primitivesCount;

    const LightObject* lights;
    size_t lightsCount;

    const LightObject* environmentMapLightObject;

    __host__ __device__
    const EnvironmentMap& getEnvironmentMap() const{
        return environmentMapLightObject->get<EnvironmentMap>();
    }
    
    
    __host__ __device__
    bool intersect(IntersectionResult& result, const Ray& ray) const{
        bool foundIntersection = false;
        for(int i = 0;i<primitivesCount;++i){
            const Primitive& prim = primitives[i];
            IntersectionResult thisResult;
            if(prim.intersect(thisResult,ray)){
                if(!foundIntersection || thisResult.distance<result.distance){
                    result = thisResult;
                    foundIntersection = true;
                }
            }
        }
        return foundIntersection;
    }

    __host__ __device__
    bool testVisibility(const VisibilityTest& test) const{
        Ray ray = test.ray;
        for(int i = 0;i<primitivesCount;++i){
            const Primitive& prim = primitives[i];
            if(prim.shape.getID() == test.sourceGeometry || prim.shape.getID() == test.targetGeometry){
                continue;
            }
            IntersectionResult thisResult;
            if(prim.shape.intersect(thisResult,ray)){
                if(test.useDistanceLimit && thisResult.distance <test.distanceLimit){
                    return false;
                }
            }
        }
        return true;
    }
};

class Scene{
public:
    std::vector<Primitive> primitivesHost;
    GpuArray<Primitive> primitivesDevice = GpuArray<Primitive>(0,true);

    std::vector<LightObject> lightsHost;
    GpuArray<LightObject> lightsDevice = GpuArray<LightObject>(0,true);

    int environmentMapIndex;

    SceneHandle getHostHandle() const{
        return {
            primitivesHost.data(),
            primitivesHost.size(),
            lightsHost.data(),
            lightsHost.size(), 
            lightsHost.data()+environmentMapIndex
        };
    }

    SceneHandle getDeviceHandle()const {
        return {
            primitivesDevice.data,
            (size_t)primitivesDevice.N,
            lightsDevice.data,
            (size_t)lightsDevice.N,
            lightsDevice.data + environmentMapIndex
        };
    }

    void copyToDevice(){
        primitivesDevice = primitivesHost;
        lightsDevice = lightsHost;
    }
    
};
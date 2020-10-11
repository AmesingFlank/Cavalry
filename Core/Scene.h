#pragma once

#include <vector>
#include "Primitive.h"
#include "Light.h"
#include "VisibilityTest.h"
#include "../Lights/LightObject.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>

struct SceneHandle{
    const Primitive* primitives;
    size_t primitivesCount;

    const LightObject* lights;
    size_t lightsCount;

    const LightObject* environmentMapLightObject;

    
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
    thrust::host_vector<Primitive> primitivesHost;
    thrust::device_vector<Primitive> primitivesDevice;

    thrust::host_vector<LightObject> lightsHost;
    thrust::device_vector<LightObject> lightsDevice;

    int environmentMapIndex;

    SceneHandle getHostHandle() const{
        return {
            thrust::raw_pointer_cast(primitivesHost.data()),
            primitivesHost.size(),
            thrust::raw_pointer_cast(lightsHost.data()),
            lightsHost.size(), 
            thrust::raw_pointer_cast(lightsHost.data())+environmentMapIndex
        };
    }

    SceneHandle getDeviceHandle()const {
        return {
            thrust::raw_pointer_cast(primitivesDevice.data()),
            primitivesDevice.size(),
            thrust::raw_pointer_cast(lightsDevice.data()),
            lightsDevice.size(),
            thrust::raw_pointer_cast(lightsDevice.data()) + environmentMapIndex
        };
    }

    void copyToDevice(){
        primitivesDevice = primitivesHost;
        lightsDevice = lightsHost;
    }
    
};
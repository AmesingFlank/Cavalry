#include "Scene.h"


SceneHandle Scene::getHostHandle() const{
    const LightObject* envMap = nullptr;
    if (environmentMapIndex >= 0) {
        envMap = (LightObject*)lightsHost.data() + environmentMapIndex;
    }
    return {
        (Primitive*)primitivesHost.data(),
        primitivesHost.size(),
        (LightObject*)lightsHost.data(),
        lightsHost.size(), 
        envMap
    };
}

SceneHandle Scene::getDeviceHandle()const {
    const LightObject* envMap = nullptr;
    if (environmentMapIndex >= 0) {
        envMap = lightsDevice.data + environmentMapIndex;
    }
    return {
        primitivesDevice.data,
        (size_t)primitivesDevice.N,
        lightsDevice.data,
        (size_t)lightsDevice.N,
        envMap
    };
}

void Scene::copyToDevice(){
    primitivesDevice = primitivesHost;
    lightsDevice = lightsHost;
}

void Scene::buildCpuReferences() {
    SceneHandle handle = getHostHandle();
    for(LightObject& light:lightsHost){
        light.buildCpuReferences(handle);
    }
};


__global__
void buildLightsGpuReferences(SceneHandle handle){
    for(int i = 0;i<handle.lightsCount;++i){
        LightObject& light = handle.lights[i];
        light.buildGpuReferences(handle);
    }
}


void Scene::buildGpuReferences() {
    SceneHandle handle = getDeviceHandle();
    buildLightsGpuReferences<<<1,1>>>(handle);
    CHECK_CUDA_ERROR("build lights gpu refs");
};

void Scene::prepareForRender() {
    for (auto& light : lightsHost) {
        light.prepareForRender();
    }
    for (auto& prim: primitivesHost) {
        prim.prepareForRender();
    }
    buildCpuReferences();
    
    copyToDevice();
    buildGpuReferences();
}
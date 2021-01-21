#include "Scene.h"
#include "../Core/Impl.h"


SceneHandle Scene::getHostHandle() const{
    const LightObject* envMap = nullptr;
    if (environmentMapIndex >= 0) {
        envMap = (LightObject*)lightsHost.data() + environmentMapIndex;
    }
    return {
        (Primitive*)primitivesHost.data(),
        primitivesHost.size(),
        (Triangle*)trianglesHost.data(),
        trianglesHost.size(),
        (LightObject*)lightsHost.data(),
        lightsHost.size(), 
        envMap,
        bvh
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
        trianglesDevice.data,
        (size_t)trianglesDevice.N,
        lightsDevice.data,
        (size_t)lightsDevice.N,
        envMap,
        bvh
    };
}

void Scene::copyToDevice(){
    primitivesDevice = primitivesHost;
    lightsDevice = lightsHost;
    trianglesDevice = trianglesHost;
}

void Scene::buildCpuReferences() {
    SceneHandle handle = getHostHandle();
    for(LightObject& light:lightsHost){
        light.buildCpuReferences(handle);
    }
    for (int i = 0; i < primitivesHost.size(); ++i) {
        Primitive& prim = primitivesHost[i];
        prim.buildCpuReferences(handle);
        prim.shape.buildCpuReferences(handle,i);
    }
    for (Triangle& triangle:trianglesHost){
        triangle.buildCpuReferences(handle);
    }
};


__global__
void buildLightsGpuReferences(SceneHandle handle){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= handle.lightsCount) return;
    
    LightObject& light = handle.lights[index];
    light.buildGpuReferences(handle);
}

__global__
void buildPrimitivesGpuReferences(SceneHandle handle) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= handle.primitivesCount) return;

    Primitive& prim = handle.primitives[index];
    prim.buildGpuReferences(handle);
    prim.shape.buildGpuReferences(handle,index);
}

__global__
void buildTrianglesGpuReferences(SceneHandle handle) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= handle.trianglesCount) return;

    Triangle& prim = handle.triangles[index];
    prim.buildGpuReferences(handle);
    
}


void Scene::buildGpuReferences() {
    SceneHandle handle = getDeviceHandle();
    int numBlocks;
    int numThreads;

    numThreads = min((int)handle.lightsCount,MAX_THREADS_PER_BLOCK);
    numBlocks = divUp(handle.lightsCount,numThreads);
    buildLightsGpuReferences<<<numBlocks,numThreads>>>(handle);
    CHECK_CUDA_ERROR("build lights gpu refs");

    numThreads = min((int)handle.primitivesCount,MAX_THREADS_PER_BLOCK);
    numBlocks = divUp(handle.primitivesCount,numThreads);
    buildPrimitivesGpuReferences <<<numBlocks, numThreads >>> (handle);
    CHECK_CUDA_ERROR("build prims gpu refs");

    numThreads = min((int)handle.trianglesCount,MAX_THREADS_PER_BLOCK);
    numBlocks = divUp(handle.trianglesCount,numThreads);
    buildTrianglesGpuReferences <<<numBlocks, numThreads >>> (handle);
    CHECK_CUDA_ERROR("build triangles gpu refs");
};

void Scene::prepareForRender() {

    for (auto& light : lightsHost) {
        light.prepareForRender();
    }
    for(int i = 0;i<primitivesHost.size();++i){
        primitivesHost[i].prepareForRender(*this,i);
    }
    
    buildCpuReferences();
    
    copyToDevice();
    buildGpuReferences();

    sceneBounds = trianglesHost[0].getBoundingBox();
    for (int i = 1; i < trianglesHost.size(); ++i) {
        //AABB temp = trianglesHost[i].getBoundingBox();
        //std::cout << "triangle bounds " << temp.minimum.x << ", " << temp.minimum.y << ", " << temp.minimum.z << ",   to" << temp.maximum.x << ", " << temp.maximum.y << ", " << temp.maximum.z << std::endl;
        sceneBounds = unionBoxes(sceneBounds, trianglesHost[i].getBoundingBox());
    }

    auto temp = sceneBounds;
    std::cout << "scene bounds " << temp.minimum.x << ", " << temp.minimum.y << ", " << temp.minimum.z << ",   to  " << temp.maximum.x << ", " << temp.maximum.y << ", " << temp.maximum.z << std::endl;

    bvh = BVH::build(trianglesDevice.data,trianglesDevice.N,sceneBounds);

    std::cout << "done preparations. TrianglesCount: " << trianglesHost.size() << std::endl;
}
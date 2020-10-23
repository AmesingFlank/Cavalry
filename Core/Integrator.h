#pragma once

#include "Scene.h"
#include "Film.h"
#include "Camera.h"
#include "Sampler.h"
#include "Color.h"
#include <memory>
#include "../Cameras/CameraObject.h"
#include "../Samplers/SamplerObject.h"

class Integrator{
public:
    virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) = 0;
    std::unique_ptr<SamplerObject> sampler;
};




template<typename Derived>
class SamplingIntegratorCPU: public Integrator{
public:
    std::unique_ptr<CameraSampler> cameraSampler;

    virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) override{
        std::vector<CameraSample> allSamples = cameraSampler->genAllSamplesCPU(camera, film);

        SceneHandle sceneHandle= scene.getHostHandle();

        SamplerObject samplerObject = *sampler;

        for (int i = 0; i < allSamples.size(); ++i) {
            const auto& sample = allSamples[i];
            Spectrum color = Derived::renderRay(sceneHandle, camera.genRay(sample), samplerObject);
            film.addSample(sample, color);

            if (i % 100 == 0) {
                std::cout << "done " << i <<" / "<<allSamples.size()<< std::endl;
            }
        }

        
        return film.readCurrentResult();
    }

};



template <typename RenderRayFn>
__global__
void renderAllSamples(CameraSample* samples, int samplesCount, SceneHandle scene, CameraObject camera, SamplerObject sampler, FilmObject film,RenderRayFn renderRay){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= samplesCount){
        return;
    }
    //samples[index].x += 1;
    //Spectrum color = make_float3(1, 0, 1);
    Ray ray = camera.genRay(samples[index]);
    Spectrum color = renderRay(scene,ray,sampler);
    film.addSample(samples[index],color);

    //samples[index].x = ray.direction.x;
}


template<typename Derived>
class SamplingIntegratorGPU: public Integrator{
public:
    std::unique_ptr<CameraSampler> cameraSampler;

    virtual RenderResult render(const Scene& scene, const CameraObject& camera, FilmObject& film) override{
        thrust::device_vector<CameraSample> allSamples = cameraSampler->genAllSamplesGPU(camera, film);

        SceneHandle sceneHandle = scene.getDeviceHandle();

        SamplerObject& samplerObject = *sampler;

        

        CameraSample* samplesPointer = thrust::raw_pointer_cast(allSamples.data());

        int samplesCount = (int)allSamples.size();
        int numThreads = min(samplesCount,MAX_THREADS_PER_BLOCK);
        int numBlocks = divUp(samplesCount,numThreads);

        CHECK_IF_CUDA_ERROR("before render all samples");
        std::cout <<sceneHandle.environmentMapLightObject<<"   "<<scene.environmentMapIndex<<"    "<<sceneHandle.lightsCount<<std::endl;
        
        renderAllSamples<<<numBlocks,numThreads>>>(samplesPointer,samplesCount,sceneHandle,camera,samplerObject.getCopyForKernel(),film.getCopyForKernel(),[]
            __device__
            (const SceneHandle& scene, const Ray& ray,SamplerObject& sampler)->Spectrum{
                return Derived::renderRay(scene,ray,sampler);
            }
        );
        

        CHECK_IF_CUDA_ERROR("after render all samples");

        CHECK_CUDA_ERROR("render all samples");
        
        return film.readCurrentResult();
    }

};

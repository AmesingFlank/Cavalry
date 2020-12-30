#include "DirectLightingIntegrator.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"
#include "../Core/Impl.h"


namespace DirectLighting {

    DirectLightingIntegrator::DirectLightingIntegrator() {

    }

    struct MaterialEvalTask {
        IntersectionResult intersection;
        Ray rayToLight;
        Spectrum incident;
        Ray exitantRay;
        float multiplier;
        Spectrum* result;
    };
    
    __device__
    static void renderRay(const SceneHandle& scene, const Ray& ray, SamplerObject& sampler, Spectrum* result, TaskQueue<MaterialEvalTask>& materialEvalQueue) {
        IntersectionResult intersection;
        scene.intersect(intersection, ray);
    
        *result = make_float3(0, 0, 0);
       
        sampler.startPixel();
        if (!intersection.intersected) {
            if (scene.hasEnvironmentMap()) {
                *result = scene.getEnvironmentMap()->EnvironmentMap::evaluateRay(ray);
            }
            return;
        }
        
    
    
        const Primitive* prim = intersection.primitive;
    
        
    
        if (prim->areaLight) {
            *result += prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::evaluateRay(ray);
        }
    
        
    
        Ray exitantRay = { intersection.position,ray.direction * -1 };
    
        int lightIndex = sampler.randInt(scene.lightsCount);
    
    
        const LightObject& light = scene.lights[lightIndex];
        Ray rayToLight;
        float probability;
        float4 randomSource = sampler.rand4();
    
        VisibilityTest visibilityTest;
        visibilityTest.sourceTriangleIndex = intersection.triangleIndex;
    
    
        Spectrum incident = light.sampleRayToPoint(intersection.position, sampler, probability, rayToLight, visibilityTest);


        if (scene.testVisibility(visibilityTest)) {
            if (probability == 0) {
                printf("probability is 0\n");
            }
            MaterialEvalTask task = {intersection,rayToLight,incident,exitantRay,(float)scene.lightsCount/probability,result};
            materialEvalQueue.push(task);
            //result += prim->material.eval(rayToLight, incident, exitantRay, intersection) * scene.lightsCount / probability;
        }
    
    }
    
    
    __global__
    void renderAllSamples(CameraSample* samples, int samplesCount, SceneHandle scene, CameraObject camera, SamplerObject sampler, Spectrum* results,TaskQueue<MaterialEvalTask> materialEvalQueue) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= samplesCount) {
            return;
        }
    
        Ray ray = camera.genRay(samples[index]);
        Spectrum* result = &results[index];
    
        renderRay(scene, ray, sampler,result,materialEvalQueue);
    
    }
    
    
    __device__
    void runMaterialEval(MaterialEvalTask& task) {
        const Primitive* prim = task.intersection.primitive;
    
        *(task.result) += prim->material.eval(task.rayToLight, task.incident, task.exitantRay, task.intersection) * task.multiplier;
    }
    
    
    __global__
    void addSamplesToFilm(FilmObject film, Spectrum* result,CameraSample* samples, int count) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        film.addSample(samples[index], result[index]);
    }



    void DirectLightingIntegrator::render(const Scene& scene, const CameraObject& camera, FilmObject& film) {

        while(!isFinished(scene,camera,film)){
            GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film);

            SceneHandle sceneHandle = scene.getDeviceHandle();

            SamplerObject& samplerObject = *sampler;


            int samplesCount = (int)allSamples.N;
            int numThreads = min(samplesCount, MAX_THREADS_PER_BLOCK);
            int numBlocks = divUp(samplesCount, numThreads);

            sampler->prepare(samplesCount);

            GpuArray<Spectrum> result(samplesCount);
            TaskQueue<DirectLighting::MaterialEvalTask> materialEvalQueue(samplesCount);

            CHECK_IF_CUDA_ERROR("before render all samples");
            DirectLighting::renderAllSamples << <numBlocks, numThreads >> >
                (allSamples.data, samplesCount, sceneHandle, camera, samplerObject.getCopyForKernel(), result.data, materialEvalQueue.getCopyForKernel());
            CHECK_IF_CUDA_ERROR("render all samples");

            materialEvalQueue.forAll(
                [] __device__
                (DirectLighting::MaterialEvalTask & task) {
                DirectLighting::runMaterialEval(task);
            }
            );

            DirectLighting::addSamplesToFilm << <numBlocks, numThreads >> > (film.getCopyForKernel(), result.data, allSamples.data, samplesCount);
            CHECK_CUDA_ERROR("add sample to film");
        }

        
    }
    
}

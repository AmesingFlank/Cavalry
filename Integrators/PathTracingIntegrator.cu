#include "PathTracingIntegrator.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"
#include "../Core/Impl.h"
#include "../Utils/Timer.h"
#include "../Utils/Utils.h"

namespace PathTracing {

    PathTracingIntegrator::PathTracingIntegrator(int maxDepth_):maxDepth(maxDepth_) {

    }

    

    struct PrimaryRayTask {
        Ray ray;
        Spectrum multiplier;
        Spectrum* result;
    };

    struct LightingTask {
        IntersectionResult intersection;
        Ray thisRay;
        Spectrum multiplier;
        Spectrum* result;
    };


    __global__
    void renderRay( SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> lightingQueue,TaskQueue<PrimaryRayTask> thisRoundRayQueue, TaskQueue<PrimaryRayTask> nextRoundRayQueue) {
        int raysCount = thisRoundRayQueue.count();
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= raysCount) {
            return;
        }


        Spectrum* result = thisRoundRayQueue.tasks.data[index].result;
        Spectrum multiplier = thisRoundRayQueue.tasks.data[index].multiplier;
        Ray thisRay = thisRoundRayQueue.tasks.data[index].ray;
        

        IntersectionResult intersection;
        scene.intersect(intersection, thisRay);

        if (!intersection.intersected) {
            if (scene.hasEnvironmentMap()) {
                *result += scene.getEnvironmentMap()->EnvironmentMap::evaluateRay(thisRay) * multiplier;
            }
            return;
        }
        
        LightingTask lightingTask = { intersection,thisRay,multiplier,result };
        lightingQueue.push(lightingTask);
        

        const Primitive* prim = intersection.primitive;


        Ray nextRay;
        float nextRayProbability;
        float3 nextDirectionLocal;
        Spectrum nextMultiplier = intersection.bsdf.sample(sampler.rand2(), nextDirectionLocal, intersection.worldToLocal(thisRay.direction * -1.f), &nextRayProbability);
        nextRay.direction = intersection.localToWorld(nextDirectionLocal);
        nextRay.origin = intersection.position + nextRay.direction * 0.0001f;
        multiplier = multiplier * nextMultiplier * abs(dot(nextRay.direction,intersection.normal)) / nextRayProbability;

        PrimaryRayTask nextTask = {nextRay,multiplier,result};
        nextRoundRayQueue.push(nextTask);
    }


    __global__
    void computeLighting(SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> tasks,int depth) {
        int tasksCount = tasks.count();
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= tasksCount) {
            return;
        }


        IntersectionResult intersection = tasks.tasks.data[index].intersection;
        Spectrum* result = tasks.tasks.data[index].result;
        Ray thisRay = tasks.tasks.data[index].thisRay;
        Spectrum multiplier = tasks.tasks.data[index].multiplier;

        const Primitive* prim = intersection.primitive;

        if (prim->areaLight) {
            if (depth == 0) {
                *result += prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::evaluateRay(thisRay) * multiplier;
            }
        }

        Ray exitantRay = { intersection.position,thisRay.direction * -1 };

        int lightIndex = sampler.randInt(scene.lightsCount);

        const LightObject& light = scene.lights[lightIndex];
        Ray rayToLight;
        float probability;
        float4 randomSource = sampler.rand4();

        VisibilityTest visibilityTest;
        visibilityTest.sourceGeometry = prim->shape.getID();


        Spectrum incident = light.sampleRayToPoint(intersection.position, sampler, probability, rayToLight, visibilityTest);

        if (scene.testVisibility(visibilityTest) && dot(rayToLight.direction, intersection.normal) > 0) {

            *result += prim->material.eval(rayToLight, incident, exitantRay, intersection) * scene.lightsCount * multiplier / probability;
        }
    }


    __global__
    void genInitialRays(CameraSample* samples, int samplesCount, CameraObject camera, Spectrum* results, TaskQueue<PrimaryRayTask> primaryRayQueue,SamplerObject sampler) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= samplesCount) {
            return;
        }

        Ray ray = camera.genRay(samples[index]);
        Spectrum* result = &results[index];
        *result = make_float3(0, 0, 0);
        Spectrum multiplier = make_float3(1, 1, 1);
        PrimaryRayTask task = { ray,multiplier,result };
        primaryRayQueue.push(task);
        sampler.startPixel();
    }

     


    __global__
    void addSamplesToFilm(FilmObject film, Spectrum* result, CameraSample* samples, int count) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        film.addSample(samples[index], result[index]);
    }



    void PathTracingIntegrator::render(const Scene& scene, const CameraObject& camera, FilmObject& film) {

        while(!isFinished( scene, camera,  film)){
            GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film);

            SceneHandle sceneHandle = scene.getDeviceHandle();
    
            SamplerObject& samplerObject = *sampler;
    
    
            int samplesCount = (int)allSamples.N;
            int numBlocks, numThreads;
            setNumBlocksThreads(samplesCount, numBlocks, numThreads);

    
            sampler->prepare(samplesCount);
    
            GpuArray<Spectrum> result(samplesCount);
            TaskQueue<LightingTask> lightingQueue(samplesCount);
    
            TaskQueue<PrimaryRayTask> primaryRayQueue0(samplesCount);
            TaskQueue<PrimaryRayTask> primaryRayQueue1(samplesCount);
    
            TaskQueue<PrimaryRayTask>* thisRoundRayQueue = &primaryRayQueue0;
            TaskQueue<PrimaryRayTask>* nextRoundRayQueue = &primaryRayQueue1;

            std::cout << numBlocks << "   " << numThreads << std::endl;
            genInitialRays << <numBlocks, numThreads >> > (allSamples.data,samplesCount,camera,result.data,thisRoundRayQueue->getCopyForKernel(), samplerObject.getCopyForKernel());
            CHECK_CUDA_ERROR("gen initial rays");

            int depth = 0;

            while (thisRoundRayQueue->count() > 0 && depth < maxDepth) {
                std::cout << "\ndoing depth " << depth << std::endl;


                thisRoundRayQueue->setNumBlocksThreads(numBlocks, numThreads);
                std::string renderRayEvent = std::string("renderRay") + std::to_string(depth);
                CHECK_IF_CUDA_ERROR("before render ray");
                Timer::getInstance().start(renderRayEvent);
                renderRay << <numBlocks, numThreads >> >
                    (sceneHandle,samplerObject.getCopyForKernel(), lightingQueue.getCopyForKernel(), thisRoundRayQueue->getCopyForKernel(),nextRoundRayQueue->getCopyForKernel());
                CHECK_CUDA_ERROR("after render ray");
                Timer::getInstance().stop(renderRayEvent);
                Timer::getInstance().printStatistics(renderRayEvent);

                thisRoundRayQueue->clear();
                
                lightingQueue.setNumBlocksThreads(numBlocks, numThreads);
                std::string lightingEvent = std::string("lighting") + std::to_string(depth);
                CHECK_IF_CUDA_ERROR("before lighting");
                Timer::getInstance().start(lightingEvent);
                computeLighting << <numBlocks, numThreads >> > (sceneHandle, samplerObject.getCopyForKernel(), lightingQueue.getCopyForKernel(),depth);
                CHECK_CUDA_ERROR("after lighting");
                Timer::getInstance().stop(lightingEvent);
                Timer::getInstance().printStatistics(lightingEvent);

                lightingQueue.clear();

                ++depth;
                std::swap(thisRoundRayQueue, nextRoundRayQueue);

            }

            setNumBlocksThreads(samplesCount, numBlocks, numThreads);

            PathTracing::addSamplesToFilm << <numBlocks, numThreads >> > (film.getCopyForKernel(), result.data, allSamples.data, samplesCount);
            CHECK_CUDA_ERROR("add sample to film");

            


        }

    }

}

#include "PathTracingIntegrator.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"
#include "../Core/Impl.h"
#include "../Utils/Timer.h"
#include "../Utils/Utils.h"

namespace PathTracing {

    PathTracingIntegrator::PathTracingIntegrator(int maxDepth_):maxDepth(maxDepth_) {

    }

    

    struct RayTask {
        Ray ray;
        Spectrum multiplier;
        Spectrum* result;
        bool shouldIncludeEmission;
    };

    struct LightingTask {
        IntersectionResult intersection;
        Ray thisRay;
        Spectrum multiplier;
        Spectrum* result;
        bool shouldIncludeEmission;
    };

    __global__
    void writeIndicesAndKeys(int N, LightingTask* tasks, int* indices, unsigned char* keys) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= N) {
            return;
        }
        
        indices[index] = index;
        keys[index] = static_cast<unsigned char>(tasks[index].intersection.primitive->material.getType());

    }

    __global__
    void applySortedIndices(int N, LightingTask* tasks, LightingTask* sortedTasks, int* sortedIndices) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= N) {
            return;
        }
        sortedTasks[index] = tasks[sortedIndices[index]];
        
    }


    void sortLightingQueue(TaskQueue<LightingTask>& queue, TaskQueue<LightingTask>& sortedQueue, SamplerObject& sampler) {
        int N = queue.count();
        if (N == 0) return;

        sortedQueue.setCount(N);

        GpuArray<int> indices(N);
        GpuArray<unsigned char> keys(N);

        int numBlocks, numThreads;
        setNumBlocksThreads(N, numBlocks, numThreads);

        writeIndicesAndKeys << <numBlocks, numThreads >> > (N, queue.tasks.data, indices.data, keys.data);
        CHECK_CUDA_ERROR("write indices and keys");

        thrust::sort_by_key(thrust::device, keys.data, keys.data+N, indices.data);

        applySortedIndices << <numBlocks, numThreads >> > (N, queue.tasks.data, sortedQueue.tasks.data,indices.data);
        CHECK_CUDA_ERROR("apply sort");

        sampler.reorderStates(indices);

    }


    __global__
    void intersectScene( SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> lightingQueue,TaskQueue<RayTask> thisRoundRayQueue, TaskQueue<RayTask> nextRoundRayQueue,int depth) {
        int raysCount = thisRoundRayQueue.count();
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= raysCount) {
            return;
        }


        RayTask& myTask = thisRoundRayQueue.tasks.data[index];
        Spectrum* result = myTask.result;
        Spectrum multiplier = myTask.multiplier;
        Ray thisRay = myTask.ray;
        

        IntersectionResult intersection;
        scene.intersect(intersection, thisRay);

        if (!intersection.intersected) {
            if (scene.hasEnvironmentMap()) {
                *result += scene.getEnvironmentMap()->EnvironmentMap::evaluateRay(thisRay) * multiplier;
            }
            return;
        }
        
        LightingTask lightingTask = { intersection,thisRay,multiplier,result, myTask.shouldIncludeEmission };
        lightingQueue.push(lightingTask);
        
    }

    __global__
    void genNextRay(SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> tasks, TaskQueue<RayTask> nextRoundRayQueue, int depth) {
        int tasksCount = tasks.count();
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= tasksCount) {
            return;
        }

        LightingTask& myTask = tasks.tasks.data[index];
        IntersectionResult intersection = myTask.intersection;
        Spectrum* result = myTask.result;
        Ray thisRay = myTask.thisRay;
        Spectrum multiplier = myTask.multiplier;

        //russian roulette
        if (depth > 3) {
            float terminationProbability = 1;
            terminationProbability = min(terminationProbability, 1 - multiplier.x);
            terminationProbability = min(terminationProbability, 1 - multiplier.y);
            terminationProbability = min(terminationProbability, 1 - multiplier.z);

            terminationProbability = max(terminationProbability, 0.05f);

            if (sampler.rand1() < terminationProbability) {
                return;
            }
            multiplier = multiplier / (1.f - terminationProbability);
        }


        const Primitive* prim = intersection.primitive;


        Ray nextRay;
        float nextRayProbability;
        float3 nextDirectionLocal;
        Spectrum nextMultiplier = intersection.bsdf.sample(sampler.rand2(), nextDirectionLocal, intersection.worldToLocal(thisRay.direction * -1.f), &nextRayProbability);
        nextRay.direction = intersection.localToWorld(nextDirectionLocal);
        nextRay.origin = intersection.position + nextRay.direction * 0.0001f;
        multiplier = multiplier * nextMultiplier * abs(dot(nextRay.direction, intersection.normal)) / nextRayProbability;
        bool nextShouldIncludeEmission = intersection.bsdf.isDelta();

        RayTask nextTask = { nextRay,multiplier,result,nextShouldIncludeEmission };
        nextRoundRayQueue.push(nextTask);
    }


    __global__
    void computeLighting(SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> tasks,int depth) {
        int tasksCount = tasks.count();
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= tasksCount) {
            return;
        }

        LightingTask& myTask = tasks.tasks.data[index];
        IntersectionResult intersection = myTask.intersection;
        Spectrum* result = myTask.result;
        Ray thisRay = myTask.thisRay;
        Spectrum multiplier = myTask.multiplier;

        const Primitive* prim = intersection.primitive;

        if (prim->areaLight) {
            if (myTask.shouldIncludeEmission) {
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
    void genInitialRays(CameraSample* samples, int samplesCount, CameraObject camera, Spectrum* results, TaskQueue<RayTask> rayQueue,SamplerObject sampler) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= samplesCount) {
            return;
        }

        Ray ray = camera.genRay(samples[index]);
        Spectrum* result = &results[index];
        *result = make_float3(0, 0, 0);
        Spectrum multiplier = make_float3(1, 1, 1);
        RayTask task = { ray,multiplier,result,true };
        rayQueue.push(task);
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
        int round = 0;

        while(!isFinished( scene, camera,  film)){
            GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film);

            SceneHandle sceneHandle = scene.getDeviceHandle();
    
            SamplerObject& samplerObject = *sampler;
    
    
            int samplesCount = (int)allSamples.N;
            int numBlocks, numThreads;
            setNumBlocksThreads(samplesCount, numBlocks, numThreads);

    
            sampler->prepare(samplesCount);
    
            GpuArray<Spectrum> result(samplesCount);

            TaskQueue<RayTask> rayQueue0(samplesCount);
            TaskQueue<RayTask> rayQueue1(samplesCount);
    
            TaskQueue<RayTask>* thisRoundRayQueue = &rayQueue0;
            TaskQueue<RayTask>* nextRoundRayQueue = &rayQueue1;

            TaskQueue<LightingTask> lightingQueue(samplesCount);
            TaskQueue<LightingTask> sortedLightingQueue(samplesCount);


            std::cout << numBlocks << "   " << numThreads << std::endl;
            genInitialRays << <numBlocks, numThreads >> > (allSamples.data,samplesCount,camera,result.data,thisRoundRayQueue->getCopyForKernel(), samplerObject.getCopyForKernel());
            CHECK_CUDA_ERROR("gen initial rays");

            int depth = 0;

            while (thisRoundRayQueue->count() > 0 && depth < maxDepth) {
                std::cout << "\ndoing depth " << depth << std::endl;


                thisRoundRayQueue->setNumBlocksThreads(numBlocks, numThreads);
                std::string intersectSceneEvent = std::string("intersectScene ") + std::to_string(round)+" " + std::to_string(depth);
                CHECK_IF_CUDA_ERROR("before intersect scene");
                Timer::getInstance().start(intersectSceneEvent);
                intersectScene << <numBlocks, numThreads >> >
                    (sceneHandle,samplerObject.getCopyForKernel(), lightingQueue.getCopyForKernel(), thisRoundRayQueue->getCopyForKernel(),nextRoundRayQueue->getCopyForKernel(),depth);
                CHECK_CUDA_ERROR("after intersectScene");
                Timer::getInstance().stop(intersectSceneEvent);
                Timer::getInstance().printStatistics(intersectSceneEvent);

                thisRoundRayQueue->clear();


                std::string sortEvent = std::string("sort queue ") + std::to_string(round) + " " + std::to_string(depth);
                Timer::getInstance().start(sortEvent);
                sortLightingQueue(lightingQueue, sortedLightingQueue, samplerObject);
                Timer::getInstance().stop(sortEvent);
                Timer::getInstance().printStatistics(sortEvent);


                sortedLightingQueue.setNumBlocksThreads(numBlocks, numThreads);
                std::string genNextRayEvent = std::string("genNext ") + std::to_string(round) + " " + std::to_string(depth);
                CHECK_IF_CUDA_ERROR("before genNext");
                Timer::getInstance().start(genNextRayEvent);
                genNextRay<< <numBlocks, numThreads >> > (sceneHandle, samplerObject.getCopyForKernel(), sortedLightingQueue.getCopyForKernel(), nextRoundRayQueue->getCopyForKernel(), depth);
                CHECK_CUDA_ERROR("after gen next round lighting");
                Timer::getInstance().stop(genNextRayEvent);
                Timer::getInstance().printStatistics(genNextRayEvent);

                sortedLightingQueue.setNumBlocksThreads(numBlocks, numThreads);
                std::string lightingEvent = std::string("lighting ") + std::to_string(round) + " " + std::to_string(depth);
                CHECK_IF_CUDA_ERROR("before lighting");
                Timer::getInstance().start(lightingEvent);
                computeLighting << <numBlocks, numThreads >> > (sceneHandle, samplerObject.getCopyForKernel(), sortedLightingQueue.getCopyForKernel(), depth);
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

            ++round;


        }

    }

}

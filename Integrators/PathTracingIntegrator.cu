#include "PathTracingIntegrator.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"



namespace PathTracing {

    PathTracingIntegrator::PathTracingIntegrator(int maxDepth_):maxDepth(maxDepth_) {

    }

    struct MaterialEvalTask {
        IntersectionResult intersection;
        Ray rayToLight;
        Spectrum incident;
        Ray exitantRay;
        float multiplier;
        Spectrum* result;
    };

    struct PrimaryRayTask {
        Ray ray;
        Spectrum multiplier;
        Spectrum* result;
    };


    __global__
    void renderRay( SceneHandle scene, SamplerObject sampler, TaskQueue<MaterialEvalTask> materialEvalQueue,TaskQueue<PrimaryRayTask> thisRoundRayQueue, TaskQueue<PrimaryRayTask> nextRoundRayQueue, int depth) {
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
                *result += scene.getEnvironmentMap()->EnvironmentMap::evaluateRay(thisRay)*multiplier;
            }
            return;
        }


        
        const Primitive* prim = intersection.primitive;

        if (prim->areaLight && depth == 0) {
            *result += prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::evaluateRay(thisRay)*multiplier;
        }


        Ray exitantRay = { intersection.position,thisRay.direction * -1 };

        int lightIndex = sampler.randInt(scene.lightsCount);

        const LightObject& light = scene.lights[lightIndex];
        Ray rayToLight;
        float probability;
        float4 randomSource = sampler.rand4();

        VisibilityTest visibilityTest;
        visibilityTest.sourceGeometry = prim->shape.getID();
        
        
        Spectrum incident = light.sampleRayToPoint(intersection.position, randomSource, probability, rayToLight, visibilityTest);

        if (scene.testVisibility(visibilityTest) && dot(rayToLight.direction, intersection.normal) > 0) {
            if (probability == 0) {
                printf("probability is 0\n");
            }
            //MaterialEvalTask task = { intersection,rayToLight,incident,exitantRay,(float)scene.lightsCount / probability,result };
            //materialEvalQueue.push(task);
            *result += prim->material.eval(rayToLight, incident, exitantRay, intersection) * scene.lightsCount*multiplier / probability;
        }

        Ray nextRay;
        float nextRayProbability;
        Spectrum nextMultiplier = prim->material.getBSDF().sample(sampler.rand2(), nextRay.direction, thisRay.direction * -1.f, &nextRayProbability);
        nextRay.origin = intersection.position + nextRay.direction * 0.0001f;
        multiplier = multiplier * nextMultiplier * abs(dot(nextRay.direction,intersection.normal)) / nextRayProbability;

        PrimaryRayTask nextTask = {nextRay,multiplier,result};
        nextRoundRayQueue.push(nextTask);

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


    __device__
    void runMaterialEval(MaterialEvalTask& task) {
        const Primitive* prim = task.intersection.primitive;

        *(task.result) += prim->material.eval(task.rayToLight, task.incident, task.exitantRay, task.intersection) * task.multiplier;
    }


    __global__
    void addSamplesToFilm(FilmObject film, Spectrum* result, CameraSample* samples, int count) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        film.addSample(samples[index], result[index]);
    }



    RenderResult PathTracingIntegrator::render(const Scene& scene, const CameraObject& camera, FilmObject& film) {

        GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film);

        SceneHandle sceneHandle = scene.getDeviceHandle();

        SamplerObject& samplerObject = *sampler;


        int samplesCount = (int)allSamples.N;
        int numThreads = min(samplesCount, MAX_THREADS_PER_BLOCK);
        int numBlocks = divUp(samplesCount, numThreads);

        sampler->prepare(samplesCount);

        GpuArray<Spectrum> result(samplesCount);
        TaskQueue<PathTracing::MaterialEvalTask> materialEvalQueue(samplesCount);

        TaskQueue<PrimaryRayTask> primaryRayQueue0(samplesCount);
        TaskQueue<PrimaryRayTask> primaryRayQueue1(samplesCount);

        TaskQueue<PrimaryRayTask>* thisRoundRayQueue = &primaryRayQueue0;
        TaskQueue<PrimaryRayTask>* nextRoundRayQueue = &primaryRayQueue1;



        genInitialRays << <numBlocks, numThreads >> > (allSamples.data,samplesCount,camera,result.data,thisRoundRayQueue->getCopyForKernel(), samplerObject.getCopyForKernel());
        CHECK_CUDA_ERROR("gen initial rays");

        int depth = 0;

        while (thisRoundRayQueue->count() > 0 && depth < maxDepth) {

            CHECK_IF_CUDA_ERROR("before render all samples");
            renderRay << <numBlocks, numThreads >> >
                (sceneHandle,samplerObject.getCopyForKernel(), materialEvalQueue.getCopyForKernel(), thisRoundRayQueue->getCopyForKernel(),nextRoundRayQueue->getCopyForKernel(),depth);
            CHECK_IF_CUDA_ERROR("render all samples");

            thisRoundRayQueue->clear();
            std::swap(thisRoundRayQueue, nextRoundRayQueue);

            materialEvalQueue.forAll(
                [] __device__
                (PathTracing::MaterialEvalTask & task) {
                PathTracing::runMaterialEval(task);
            }
            );

            ++depth;

        }

        PathTracing::addSamplesToFilm << <numBlocks, numThreads >> > (film.getCopyForKernel(), result.data, allSamples.data, samplesCount);
        CHECK_CUDA_ERROR("add sample to film");

        return film.readCurrentResult();
    }

}

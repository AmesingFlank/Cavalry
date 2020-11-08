#include "DirectLightingGPUIntegrator.h"
#include "../Samplers/SimpleSamplerGPU.h"

DirectLightingGPUIntegrator::DirectLightingGPUIntegrator() {

}

struct MaterialEvalTask {
    IntersectionResult intersection;
    Ray rayToLight;
    Spectrum incident;
    Ray exitantRay;
    float probability;
};

__device__
static Spectrum renderRay(const SceneHandle& scene, const Ray& ray, SamplerObject& sampler) {
    IntersectionResult intersection;
    scene.intersect(intersection, ray);


    if (!intersection.intersected) {
        if (scene.hasEnvironmentMap()) {
            return scene.getEnvironmentMap()->EnvironmentMap::evaluateRay(ray);
        }
        return make_float3(0, 0, 0);
    }


    Spectrum result = make_float3(0, 0, 0);


    const Primitive* prim = intersection.primitive;

    if (prim->areaLight) {
        result += prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::evaluateRay(ray);
    }

    Ray exitantRay = { intersection.position,ray.direction * -1 };

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

        result += prim->material.eval(rayToLight, incident, exitantRay, intersection) * scene.lightsCount / probability;
    }
    
    return result;
}


__global__
void renderAllSamples(CameraSample* samples, int samplesCount, SceneHandle scene, CameraObject camera, SamplerObject sampler, FilmObject film) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= samplesCount) {
        return;
    }

    Ray ray = camera.genRay(samples[index]);
    Spectrum color = renderRay(scene, ray, sampler);
    film.addSample(samples[index], color);

}


RenderResult DirectLightingGPUIntegrator::render(const Scene& scene, const CameraObject& camera, FilmObject& film) {

    GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film);

    SceneHandle sceneHandle = scene.getDeviceHandle();

    SamplerObject& samplerObject = *sampler;


    int samplesCount = (int)allSamples.N;
    int numThreads = min(samplesCount, MAX_THREADS_PER_BLOCK);
    int numBlocks = divUp(samplesCount, numThreads);

    GpuArray<CameraSample> results(samplesCount);

    renderAllSamples << <numBlocks, numThreads >> > (allSamples.data, samplesCount, sceneHandle, camera, samplerObject.getCopyForKernel(), film.getCopyForKernel());
    CHECK_CUDA_ERROR("render all samples");



    return film.readCurrentResult();
}
#include "ReinforcementLearningPathTracing.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"
#include "../Core/Impl.h"
#include "../Utils/Timer.h"
#include "../Utils/Utils.h"

namespace ReinforcementLearningPathTracing {

    struct QEntry {
        static constexpr int NUM_X = 16;
        static constexpr int NUM_Y = 8;
        static constexpr int NUM_XY = NUM_X * NUM_Y;

        __host__ __device__
        static constexpr float INV_NUM_X() {
            return 1.f / (float)NUM_X;
        }

        __host__ __device__
        static constexpr float INV_NUM_Y() {
            return 1.f / (float)NUM_Y;
        }

        float Q[NUM_XY];

        float newQ[NUM_XY];
        float proposalCount[NUM_XY];
        float totalProposalCount[NUM_XY];

        __device__
        float defaultQ(int cellIndex)const {
            return 1;
        }

        __device__
        QEntry() {
            for (int i = 0; i < NUM_XY; ++i) {
                Q[i] = defaultQ(i);
                totalProposalCount[i] = 0;
            }
        }

        __device__
        float alpha(int cellIndex) const {
            //printf("computing alpha[%d]:  %f %f\n", cellIndex,proposalCount[cellIndex],totalProposalCount[cellIndex]);
            if (proposalCount[cellIndex] == 0) {
                return 0;
            }
            return (proposalCount[cellIndex]) / (totalProposalCount[cellIndex] + proposalCount[cellIndex]);
        }

        __device__
        float averageQ() {
            float sumQ = 0;
            int count = 0;
            for (int i = 0; i < NUM_XY; ++i) {
                if (totalProposalCount[i] > 0) {
                    sumQ += Q[i];
                    count += 1;
                }
            }
            if (count == 0) {
                return defaultQ(0);
            }
            float avg = sumQ / (float)count;
            for (int i = 0; i < NUM_XY; ++i) {
                if (totalProposalCount[i] == 0) {
                    Q[i] = avg;
                    //printf("updaing avg%f\n", avg);
                }
            }
            return avg;
        }

        __device__
        float sumQ() {
            float sumQ = 0;
            for (int i = 0; i < NUM_XY; ++i) {
                sumQ += Q[i];
            }
            return sumQ;
        }


        // should only be called by one thread for each cellIndex
        __device__
        void prepareForUpdateQ(int cellIndex) {
            newQ[cellIndex] = 0;
            proposalCount[cellIndex] = 0;
        }

        __device__
        void proposeNextQ(float QVal, int cellIndex) {
            atomicAdd(&(newQ[cellIndex]), QVal);
            atomicAdd(&(proposalCount[cellIndex]), 1);
            //printf("proposing %d %f\n", cellIndex, QVal);
        }

        // should only be called by one thread for each cellIndex
        __device__
        void finishUpdateQ(int cellIndex) {
            float a = alpha(cellIndex);
            if (a == 0) {
                return;
            }
            float updatedQ = Q[cellIndex] * (1.f - a) + a * newQ[cellIndex] / proposalCount[cellIndex];
            //printf("Q[%d]:   old:%f  new:%f  alpha:%f \n", cellIndex, Q[cellIndex], updatedQ, a);
            Q[cellIndex] = updatedQ;
            totalProposalCount[cellIndex] += proposalCount[cellIndex];
        }


        static __host__ __device__  int dirToCellIndex(float3 dir) {
            float u = dir.z;
            float y = (u + 1.f) / 2.f;
            int thetaIndex = clampF((int)(y * QEntry::NUM_Y), 0, QEntry::NUM_Y - 1);

            dir.x /= sqrt(1.f - u * u);
            dir.y /= sqrt(1.f - u * u);

            float v = acos(dir.x);
            if (dir.y < 0) {
                v = (2.f * M_PI) - v;
            }
            v = v / (2.f * M_PI);
            int phiIndex = clampF((int)(v * QEntry::NUM_X), 0, QEntry::NUM_X - 1);
            return thetaIndex * NUM_X + phiIndex;
        }

        __device__
        float3 sampleDirectionInCell(float2 randomSource, int cellIndex) const
        {
            int thetaIdx = cellIndex / NUM_X;
            int phiIdx = cellIndex % NUM_X;
            float u = ((float)thetaIdx + randomSource.x) * INV_NUM_Y();
            u = u * 2 - 1.f;
            float v = ((float)phiIdx + randomSource.y) * INV_NUM_X();

            float xyScale = sqrt(1.0f - u * u);
            float phi = 2 * M_PI * v;

            float3 dir = make_float3(
                xyScale * cos(phi),
                xyScale * sin(phi),
                u);
            return dir;
        }
    };


#define Q_TABLE_SIZE 32

    using QDistribution = FixedSizeDistribution1D<QEntry::NUM_XY>;

    RLPTIntegrator::RLPTIntegrator(int maxDepth_):maxDepth(maxDepth_) {

    }

    struct QEntryInfo {
        int entryIndex = -1;
        int cellIndex = -1;
    };

    __device__
    int findQEntry(const AABB& sceneBounds, float3 position){
        float3 sceneSize = sceneBounds.maximum-sceneBounds.minimum;
        float3 entrySize = sceneSize / make_float3(Q_TABLE_SIZE,Q_TABLE_SIZE,Q_TABLE_SIZE);
        int3 entryIndex;
        position -= sceneBounds.minimum;
        entryIndex.x = clampF((int)(position.x / entrySize.x),0,Q_TABLE_SIZE-1);
        entryIndex.y = clampF((int)(position.y / entrySize.y),0,Q_TABLE_SIZE-1);
        entryIndex.z = clampF((int)(position.z / entrySize.z),0,Q_TABLE_SIZE-1);
        return entryIndex.x * Q_TABLE_SIZE * Q_TABLE_SIZE + entryIndex.y * Q_TABLE_SIZE + entryIndex.z;
    }

    struct RayTask {
        SamplingState samplingState;
        Ray ray;
        Spectrum multiplier;
        Spectrum* result;
        float surfacePDF; // if the ray is generated by sampling a BSDF, this is the PDF of that sample. This is needed for MIS
        bool sampledFromDeltaBSDF;
        QEntryInfo previousQEntry;
    };

    struct LightingTask {
        SamplingState samplingState;
        IntersectionResult intersection;
        Ray thisRay;
        Spectrum multiplier;
        Spectrum* result;
        float surfacePDF;
        bool sampledFromDeltaBSDF;
        QEntryInfo previousQEntry;
    };

    struct LightingResult {
        Spectrum indirectLightingContrib;
        float3 rayToLightDirection;
        float directLightingContrib;
        float directLightingImmediate;
        float lightPDF;
    };

    struct NextRayInfo {
        float3 dir;
        float surfacePDF;
        int cellIndex;
        bool valid; // whether or not a valid ray dir has been computed during computeQDist
    };

    __device__
    inline float misPowerHeuristic(float pdfA, float pdfB) {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    
    // use q entry index as part of the key for sorting. this improves locality.
    __global__
    void writeIndicesAndKeys(int N, LightingTask* tasks, int* indices, int* keys,AABB sceneBounds) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= N) {
            return;
        }
        
        indices[index] = index;
        int materialID = static_cast<int>(tasks[index].intersection.primitive->material.getType());
        int entryIndex = findQEntry(sceneBounds, tasks[index].intersection.position);

        int key = Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE * materialID + entryIndex;
        keys[index] = key;
    }

    __global__
    void applySortedIndices(int N,int* sortedIndices, LightingTask* lightTasks, LightingTask* lightTasksCopy) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= N) {
            return;
        }
        lightTasksCopy[index] = lightTasks[sortedIndices[index]];        
    }

    // sort the lighting queue using material and q entry id as key.
    void sortLightingQueue(TaskQueue<LightingTask>& lightQueue, TaskQueue<LightingTask>& lightQueueCopy, AABB sceneBounds) {
        int N = lightQueue.count();
        if (N == 0) return;

        lightQueueCopy.setCount(N);

        GpuArray<int> indices(N);
        GpuArray<int> keys(N);

        int numBlocks, numThreads;
        setNumBlocksThreads(N, numBlocks, numThreads);

        writeIndicesAndKeys << <numBlocks, numThreads >> > (N, lightQueue.tasks.data, indices.data, keys.data,sceneBounds);
        CHECK_CUDA_ERROR("write indices and keys");

        thrust::stable_sort_by_key(thrust::device, keys.data, keys.data+N, indices.data);

        applySortedIndices << <numBlocks, numThreads >> > (N,indices.data, lightQueue.tasks.data, lightQueueCopy.tasks.data);
        CHECK_CUDA_ERROR("apply sort");
        std::swap(lightQueue.tasks.data, lightQueueCopy.tasks.data);

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
        
        LightingTask lightingTask = { myTask.samplingState, intersection,thisRay,multiplier,result,myTask.surfacePDF,myTask.sampledFromDeltaBSDF,myTask.previousQEntry };
        lightingQueue.push(lightingTask);
        
    }

    __global__
    void genNextRay(SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> tasks, TaskQueue<RayTask> nextRoundRayQueue, int depth, GpuArray<QEntry> QTable,GpuArray<NextRayInfo> nextRayInfos) {
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

            if (sampler.rand1(myTask.samplingState) < terminationProbability) {
                return;
            }
            multiplier = multiplier / (1.f - terminationProbability);
        }

        const Primitive* prim = intersection.primitive;         

        Ray nextRay;
        float nextRayProbability;

        float3 tangent0, tangent1;
        intersection.findTangents(tangent0, tangent1);

        float3 exitantDir = thisRay.direction * -1.f;

        QEntryInfo entryInfo;
        entryInfo.entryIndex = findQEntry(scene.sceneBounds, intersection.position);

        Spectrum nextMultiplier;

        NextRayInfo& info = nextRayInfos.data[index];

        if (intersection.bsdf.isDelta() || intersection.bsdf.isAlmostDelta() ||  !info.valid) {
            float3 nextDirectionLocal;
            nextMultiplier = intersection.bsdf.sample(sampler.rand2(myTask.samplingState), nextDirectionLocal, intersection.worldToLocal(exitantDir,tangent0,tangent1), &nextRayProbability);
            nextRay.direction = intersection.localToWorld(nextDirectionLocal);
            nextRay.origin = intersection.position + nextRay.direction * 0.001f;
            entryInfo.cellIndex = QEntry::dirToCellIndex(nextRay.direction);
        }
        else {
            entryInfo.cellIndex = info.cellIndex;
            nextRayProbability = info.surfacePDF;
            nextRay.direction = info.dir;
            nextRay.origin = intersection.position + nextRay.direction * 0.001f;

            nextMultiplier = intersection.bsdf.eval(intersection.worldToLocal(nextRay.direction, tangent0, tangent1), intersection.worldToLocal(exitantDir, tangent0, tangent1));
        }

        if (isAllZero(nextMultiplier)) {
            multiplier = make_float3(0, 0, 0);
        }
        else {
            multiplier = multiplier * nextMultiplier * abs(dot(nextRay.direction, intersection.normal)) / nextRayProbability;
        }

        RayTask nextTask = { myTask.samplingState, nextRay,multiplier,result,nextRayProbability, intersection.bsdf.isDelta(),entryInfo };
        nextRoundRayQueue.push(nextTask);
    }

    __global__
    void prepareLightingTraining(SceneHandle scene, GpuArray<float> lightSamplingNewResults, GpuArray<float> lightSamplingNewCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= lightSamplingNewResults.N) {
            return;
        }
        lightSamplingNewResults.data[index] = 0;
        lightSamplingNewCount.data[index] = 0;
    }

    __global__
    void computeLighting(SceneHandle scene, SamplerObject sampler, TaskQueue<LightingTask> tasks, int depth,GpuArray<QEntry> QTable,GpuArray<LightingResult> results,
        GpuArray<float> lightSamplingDist, GpuArray<float> lightSamplingNewResults,  GpuArray<float> lightSamplingNewCount) {
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
        QEntryInfo& previousQEntry = myTask.previousQEntry;

        const Primitive* prim = intersection.primitive;

        Spectrum directLightingContrib = make_float3(0, 0, 0);

        if (prim->areaLight) {
            if (myTask.sampledFromDeltaBSDF) {
                // then don't apply MIS, because the sampleRayToPoint call had a 0 probability of finding any radiance;
                *result += prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::evaluateRay(thisRay,intersection) * multiplier;
            }
            else {
                if (depth > 0) {
                    float surfacePDF = myTask.surfacePDF;
                    float lightPDF = prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::sampleRayToPointPdf(thisRay, intersection);
                    float misWeight = misPowerHeuristic(surfacePDF, lightPDF);
                    if (isfinite(misWeight)) {
                        Spectrum emmited = prim->areaLight->get<DiffuseAreaLight>()->DiffuseAreaLight::evaluateRay(thisRay, intersection);
                        directLightingContrib = emmited;
                        *result += emmited * multiplier * misWeight;
                    }
                }
            }
        }

        int thisEntryIndex = findQEntry(scene.sceneBounds, intersection.position);
        Distribution1D lightDist(scene.lightsCount, lightSamplingDist.data + thisEntryIndex * scene.lightsCount);

        float lightSelectionProbability;
        int lightIndex = lightDist.sample(sampler.rand1(myTask.samplingState), lightSelectionProbability);

        const LightObject& light = scene.lights[lightIndex];
        Ray rayToLight;
        float4 randomSource = sampler.rand4(myTask.samplingState);

        VisibilityTest visibilityTest;
        visibilityTest.sourceMeshIndex = intersection.primitive->shape.meshIndex;

        float lightPDF;
        Spectrum incident = light.sampleRayToPoint(intersection.position, sampler,myTask.samplingState, lightPDF, rayToLight, visibilityTest, nullptr);

        if (!(scene.testVisibility(visibilityTest) && isfinite(lightPDF))) {
            // then light is occluded. But still call materialEval in order to update Q.
            incident = make_float3(0, 0, 0); 
            lightPDF = 1;
        }
        Ray exitantRay = { intersection.position,thisRay.direction * -1 };
        Spectrum indirectLightingContrib = intersection.primitive->material.eval(rayToLight, incident, exitantRay, intersection);
        indirectLightingContrib *= 1.f / (lightPDF * lightSelectionProbability);

        atomicAdd(lightSamplingNewResults.data + thisEntryIndex * scene.lightsCount + lightIndex, luminance(incident));
        atomicAdd(lightSamplingNewCount.data + thisEntryIndex * scene.lightsCount + lightIndex, 1.f);


        results.data[index].indirectLightingContrib = indirectLightingContrib;
        results.data[index].lightPDF = lightPDF;
        results.data[index].directLightingContrib = luminance(directLightingContrib);
        results.data[index].rayToLightDirection = rayToLight.direction;
        results.data[index].directLightingImmediate = luminance(incident);// *1.f / (lightPDF * lightSelectionProbability);

    }

    __global__
    void finishLightingTraining(SceneHandle scene, GpuArray<float> lightSamplingDist, GpuArray<float> lightSamplingResults, GpuArray<float> lightSamplingCount, GpuArray<float> lightSamplingNewResults, GpuArray<float> lightSamplingNewCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= Q_TABLE_SIZE * Q_TABLE_SIZE* Q_TABLE_SIZE) {
            return;
        }
        int lightsCount = scene.lightsCount;
        float sumResult = 0;
        float validCount = 0;
        Distribution1D lightDist(lightsCount, lightSamplingDist.data + index*lightsCount);
        for (int i = 0; i < lightsCount; ++i) {
            float& count = lightSamplingCount.data[index * lightsCount + i];
            float& newCount = lightSamplingNewCount.data[index * lightsCount + i];
            float& result = lightSamplingResults.data[index * lightsCount + i];
            float& newResult = lightSamplingNewResults.data[index * lightsCount + i];

            float oldResult = result;
            
            if (newCount > 0) {
                result = (count / (count + newCount)) * result + (newCount / (count + newCount)) * (newResult / newCount);
                count += newCount;
                sumResult += result;
                validCount += 1;
            }

            newCount = 0;
            newResult = 0;
        }
        if (sumResult > 0) {
            float avgResult = sumResult / (float)validCount;

            float sumDensity = 0;
            for (int i = 0; i < lightsCount; ++i) {
                float& count = lightSamplingCount.data[index * lightsCount + i];
                float& result = lightSamplingResults.data[index * lightsCount + i];
                if (result > 0) {
                    sumDensity += result;
                    //printf("hey %f %d \n", result,index);
                }
                else if (count > 0) {
                    sumDensity += avgResult / (float) count;
                }
                else {
                    sumDensity += avgResult;
                }
                lightDist.cdf[i] = sumDensity;
            }
            for (int i = 0; i < lightsCount; ++i) {
                lightDist.cdf[i] /= sumDensity;
            }
        }
    }

    __global__
    void computeQDistributions(SceneHandle scene,TaskQueue<LightingTask> lightingTasks, GpuArray<LightingResult> lightingResults, GpuArray<QEntry> QTable,SamplerObject sampler,GpuArray<NextRayInfo> nextRayInfos) {
        int tasksCount = lightingTasks.count();
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= tasksCount) {
            return;
        }

        LightingTask& myTask = lightingTasks.tasks.data[index];
        IntersectionResult& intersection = myTask.intersection;
        Spectrum* result = myTask.result;
        Spectrum& multiplier = myTask.multiplier;
        QEntryInfo previousQEntry = myTask.previousQEntry;

        float3 incidentDir = lightingResults.data[index].rayToLightDirection;
        float3 exitantDir = myTask.thisRay.direction * -1.f;

        

        float sumWeightedQ = 0.f; // This will be used for two different things: updating Q table, and computing surfacePDF (for MIS);

        int thisQEntryIndex = findQEntry(scene.sceneBounds, intersection.position);
        QEntry& thisEntry = QTable.data[thisQEntryIndex];

        float3 tangent0, tangent1;
        intersection.findTangents(tangent0, tangent1);
        float3 exitantLocal = intersection.worldToLocal(exitantDir, tangent0, tangent1);

        QDistribution dist;

        float2 positionInCell = sampler.rand2(myTask.samplingState);

        for (int cellIndex = 0; cellIndex < QEntry::NUM_XY; ++cellIndex) {

            float3 dir = thisEntry.sampleDirectionInCell(positionInCell, cellIndex);
            float3 incidentLocal = intersection.worldToLocal(dir, tangent0, tangent1);

            float scattering = luminance(intersection.bsdf.eval(incidentLocal, exitantLocal));
            float thisDirQ = abs(dot(dir, intersection.normal)) * scattering * thisEntry.Q[cellIndex];
            sumWeightedQ += thisDirQ;
            dist.cdf[cellIndex] = sumWeightedQ;
        }

        for (int cellIndex = 0; cellIndex < QEntry::NUM_XY; ++cellIndex) {
            dist.cdf[cellIndex] /= sumWeightedQ;
        }

        sumWeightedQ *= (4.f * M_PI / (float)QEntry::NUM_XY);

        // updated Q
        if (previousQEntry.entryIndex != -1) {
            // update q table
            auto& lightingRes = lightingResults.data[index];
            float proposal =  sumWeightedQ + lightingRes.directLightingContrib + luminance(lightingRes.indirectLightingContrib);
            QTable.data[previousQEntry.entryIndex].proposeNextQ(proposal, previousQEntry.cellIndex);
            if ( lightingRes.directLightingContrib > sumWeightedQ) {
                //printf("contribs  %f, %f,  %f\n", luminance(lightingRes.directLightingContrib), luminance(lightingRes.indirectLightingContrib), sumWeightedQ);
            }
        }

        // compute MIS
        int rayToLightCellIndex = QEntry::dirToCellIndex(incidentDir);
        float surfacePDF = 0;

        float3 incidentLocal = intersection.worldToLocal(incidentDir, tangent0, tangent1);
        float materialPDF = intersection.bsdf.pdf(incidentLocal, exitantLocal);

        if (sumWeightedQ != 0) {
            surfacePDF = dist.cdf[rayToLightCellIndex];
            if (rayToLightCellIndex > 0) {
                surfacePDF -= dist.cdf[rayToLightCellIndex - 1];
            }
            surfacePDF = (QEntry::NUM_XY * surfacePDF / (4 * M_PI)); // Solid angle probability
        }
        if(sumWeightedQ==0 || surfacePDF == 0 || intersection.bsdf.isDelta() || intersection.bsdf.isAlmostDelta()){
            // then we shouldn't trust surfacePDF
            // either ecause we haven't done enough learning
            surfacePDF = materialPDF;
        }
        float lightPDF = lightingResults.data[index].lightPDF;
        float misWeight = misPowerHeuristic(lightPDF, surfacePDF);

        *result += lightingResults.data[index].indirectLightingContrib * multiplier * misWeight;

        // update this entry as well, using direct lighting sampled.
        if ( lightingResults.data[index].directLightingImmediate > 0) {
            thisEntry.proposeNextQ(lightingResults.data[index].directLightingImmediate, rayToLightCellIndex);
            //printf("proposing %f %f \n", luminance(lightingResults.data[index].directLightingHere), thisEntry.Q[rayToLightCellIndex]);
        }

        // sample next ray Dir
        float& nextRayProbability = nextRayInfos.data[index].surfacePDF;
        if (sumWeightedQ != 0) {
            nextRayInfos.data[index].cellIndex = dist.sample(sampler.rand1(myTask.samplingState), nextRayProbability);
            //nextRayInfos.data[index].cellIndex = dist.mode(nextRayProbability);

            if (index == 100) {
                //printf("prob %f   \n", nextRayProbability);
            }
            nextRayProbability = (QEntry::NUM_XY * nextRayProbability / (4 * M_PI)); // Solid angle probability
            nextRayInfos.data[index].dir = thisEntry.sampleDirectionInCell(positionInCell, nextRayInfos.data[index].cellIndex);
            nextRayInfos.data[index].valid = true;
        }
        else {
            nextRayInfos.data[index].valid = false;
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
        QEntryInfo nullEntry = { -1,-1};

        RayTask task = {samples[index].samplingState,ray,multiplier,result,1,true,nullEntry };
        rayQueue.push(task);
    }

     


    __global__
    void addSamplesToFilm(Film film, Spectrum* result, CameraSample* samples, int count) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        film.addSample(samples[index], result[index]);
    }

    template<typename TaskType>
    __global__
    void findMaxDimension(TaskType* tasks, int N, int* maxDimension) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= N) {
            return;
        }
        int dim = tasks[index].samplingState.dimension;
        atomicMax(maxDimension, dim);
    }

    template<typename TaskType>
    __global__
    void setMaxDimension(TaskType* tasks, int N, int* maxDimension) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= N) {
            return;
        }
        tasks[index].samplingState.dimension = *maxDimension;
    }

    template<typename TaskType>
    void syncDimension(TaskType* tasks, int N, GpuArray<int>& maxDimension) {
        int numBlocks, numThreads;
        setNumBlocksThreads(N, numBlocks, numThreads);

        maxDimension.set(0, 0);

        findMaxDimension << <numBlocks, numThreads >> > (tasks, N, maxDimension.data);
        CHECK_CUDA_ERROR("write max halton dimension");
        //std::cout << "maxDimension: " << maxDimension.get(0) << std::endl;

        setMaxDimension << <numBlocks, numThreads >> > (tasks, N, maxDimension.data);
        CHECK_CUDA_ERROR("sync halton dimension");
    }

    __global__
    void initialiseQTable(GpuArray<QEntry> QTable){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= QTable.N) {
            return;
        }
        QTable.data[index] = QEntry();
    }

    __global__
    void initialiseLightSamplingDist(SceneHandle scene, GpuArray<float> lightSamplingDist) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE) {
            return;
        }
        int lightsCount = scene.lightsCount;
        Distribution1D lightDist(lightsCount, lightSamplingDist.data + index * lightsCount);
        for (int i = 0; i < lightsCount; ++i) {
            lightDist.cdf[i] = ((float)i + 1.f) / (float)lightsCount;
        }
        
    }

    __global__
    void prepareForUpdateQ(GpuArray<QEntry> QTable){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= QTable.N*QEntry::NUM_XY) {
            return;
        }
        int entryIndex = index / QEntry::NUM_XY;
        int cellIndex = index % QEntry::NUM_XY;
        QTable.data[entryIndex].prepareForUpdateQ(cellIndex);
    }

    __global__
    void finishUpdateQ(GpuArray<QEntry> QTable){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= QTable.N*QEntry::NUM_XY) {
            return;
        }
        int entryIndex = index / QEntry::NUM_XY;
        int cellIndex = index % QEntry::NUM_XY;
        QTable.data[entryIndex].finishUpdateQ(cellIndex);
    }

    __global__
    void averageQ(GpuArray<QEntry> QTable) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= QTable.N ) {
            return;
        }
        
        QTable.data[index].averageQ();
    }


    void debugPrintQTable(const GpuArray<QEntry>& QTable) {
        int size = Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE;
        std::vector<QEntry> table = QTable.toVector();
        for (int i = 0; i <size; ++i) {
            QEntry entry = QTable.get(i);
            bool hasAnyProposals = false;
            for (int j = 0; j < QEntry::NUM_XY; ++j) {
                if (entry.totalProposalCount[j] > 0) {
                    hasAnyProposals = true;
                    break;
                }
            }
            if (!hasAnyProposals) {
                std::cout << "Entry: " << i << "  no info"<<std::endl;
                continue;
            }
            std::cout << "Entry: " << i << std::endl;
            for (int y = 0; y < QEntry::NUM_Y; y++){
                std::cout << "y=" << y << "   ";
                for (int x = 0; x < QEntry::NUM_X; x++){
                    int thisIndex = x + y * QEntry::NUM_X;
                    printf("%f(%d)\t", entry.Q[thisIndex],(int)entry.totalProposalCount[thisIndex]);
                    //std::cout << entry.Q[x + y * QEntry::NUM_X] << "  ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
    }



    void RLPTIntegrator::render(const Scene& scene, const CameraObject& camera, Film& film) {
        SceneHandle sceneHandle = scene.getDeviceHandle();
        SamplerObject& samplerObject = *sampler;

        int lightsCount = scene.lightsHost.size();
        GpuArray<QEntry> QTable(Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE, false);
        GpuArray<float> lightSamplingDist(Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE * lightsCount, false);

        GpuArray<float> lightSamplingResults(Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE * lightsCount, false);
        GpuArray<float> lightSamplingNewResults(Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE * lightsCount, false);
        GpuArray<float> lightSamplingCount(Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE * lightsCount, false);
        GpuArray<float> lightSamplingNewCount(Q_TABLE_SIZE * Q_TABLE_SIZE * Q_TABLE_SIZE * lightsCount, false);


        int bytesNeededPerThread = sizeof(CameraSample) + sampler->bytesNeededPerThread() + sizeof(Spectrum) + sizeof(RayTask)*2 + sizeof(LightingTask)*2 + sizeof(LightingResult)+sizeof(NextRayInfo)+4*sizeof(int) ;
        std::cout<<"Running RL Path Tracing Integrator. Bytes needed per thread: "<<bytesNeededPerThread<<std::endl;

        int numBlocks, numThreads;
        setNumBlocksThreads(QTable.N, numBlocks, numThreads);
        initialiseQTable<<<numBlocks,numThreads>>>(QTable.getCopyForKernel());

        setNumBlocksThreads(QTable.N, numBlocks, numThreads);
        initialiseLightSamplingDist <<<numBlocks, numThreads >>> (sceneHandle,lightSamplingDist.getCopyForKernel());

        int round = 0;

        GpuArray<int> maxDimension(1);

        while(!isFinished( scene, camera,  film)){
            GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film, bytesNeededPerThread);
        
            int samplesCount = (int)allSamples.N;
            setNumBlocksThreads(samplesCount, numBlocks, numThreads);

            sampler->prepare(samplesCount);

            //debugTestEntrySampling << <1, 1 >> > (samplerObject);
            //SIGNAL_ERROR("done testing\n");
    
            GpuArray<Spectrum> result(samplesCount);

            TaskQueue<RayTask> rayQueue0(samplesCount);
            TaskQueue<RayTask> rayQueue1(samplesCount);
    
            TaskQueue<RayTask>* thisRoundRayQueue = &rayQueue0;
            TaskQueue<RayTask>* nextRoundRayQueue = &rayQueue1;

            TaskQueue<LightingTask> lightingQueue(samplesCount);
            TaskQueue<LightingTask> lightingQueueCopy(samplesCount);

            GpuArray<LightingResult> lightingResults(samplesCount);
            GpuArray<NextRayInfo> nextRayInfos(samplesCount);

            int QCellsCount = QTable.N * QEntry::NUM_XY;

            std::cout << numBlocks << "   " << numThreads << std::endl;
            genInitialRays << <numBlocks, numThreads >> > (allSamples.data,samplesCount,camera,result.data,thisRoundRayQueue->getCopyForKernel(), samplerObject.getCopyForKernel());
            CHECK_CUDA_ERROR("gen initial rays");

            int depth = 0;

            while (thisRoundRayQueue->count() > 0 && depth < maxDepth) {
                //std::cout << "\ndoing depth " << depth << std::endl;

                //if(depth>=2)  debugPrintQTable(QTable);

                thisRoundRayQueue->setNumBlocksThreads(numBlocks, numThreads);
                std::string intersectSceneEvent = std::string("intersectScene ") + std::to_string(round)+" " + std::to_string(depth);
                Timer::getInstance().timedRun(intersectSceneEvent, [&](){
                    intersectScene << <numBlocks, numThreads >> >
                        (sceneHandle, samplerObject.getCopyForKernel(), lightingQueue.getCopyForKernel(), thisRoundRayQueue->getCopyForKernel(), nextRoundRayQueue->getCopyForKernel(), depth);
                });
                

                thisRoundRayQueue->clear();


                if (lightingQueue.count() > 0) {
                    std::string sortEvent = std::string("sort queue ") + std::to_string(round) + " " + std::to_string(depth);
                    Timer::getInstance().timedRun(sortEvent, [&]() {
                        sortLightingQueue(lightingQueue, lightingQueueCopy, scene.sceneBounds);
                    });

                    setNumBlocksThreads(QCellsCount, numBlocks, numThreads);
                    prepareForUpdateQ << <numBlocks, numThreads >> > (QTable.getCopyForKernel());
                    CHECK_CUDA_ERROR("prepare update q");

                    setNumBlocksThreads(lightsCount * QTable.N, numBlocks, numThreads);
                    prepareLightingTraining << <numBlocks, numThreads >> > (sceneHandle, lightSamplingNewResults.getCopyForKernel(), lightSamplingNewCount.getCopyForKernel());
                    CHECK_CUDA_ERROR("prepare lighting training");

                    syncDimension(lightingQueue.tasks.data, lightingQueue.count(), maxDimension);

                    lightingQueue.setNumBlocksThreads(numBlocks, numThreads);
                    std::string lightingEvent = std::string("lighting ") + std::to_string(round) + " " + std::to_string(depth);
                    Timer::getInstance().timedRun(lightingEvent, [&]() {
                        computeLighting << <numBlocks, numThreads >> > (sceneHandle, samplerObject.getCopyForKernel(), lightingQueue.getCopyForKernel(), depth,QTable.getCopyForKernel(), lightingResults.getCopyForKernel(), lightSamplingDist.getCopyForKernel(), lightSamplingNewResults.getCopyForKernel(), lightSamplingNewCount.getCopyForKernel());
                    });


                    std::string QEvent = std::string("compute Q ") + std::to_string(round) + " " + std::to_string(depth);
                    Timer::getInstance().timedRun(QEvent, [&]() {
                        computeQDistributions << <numBlocks, numThreads >> > (sceneHandle, lightingQueue.getCopyForKernel(), lightingResults.getCopyForKernel(), QTable.getCopyForKernel(), samplerObject.getCopyForKernel(), nextRayInfos.getCopyForKernel());
                    });


                    setNumBlocksThreads(QCellsCount,numBlocks,numThreads);
                    finishUpdateQ<<<numBlocks,numThreads>>>(QTable.getCopyForKernel());
                    CHECK_CUDA_ERROR("finish update q");

                    setNumBlocksThreads(QTable.N, numBlocks, numThreads);
                    averageQ << <numBlocks, numThreads >> > (QTable.getCopyForKernel());
                    CHECK_CUDA_ERROR("average q");

                    setNumBlocksThreads(lightsCount * QTable.N, numBlocks, numThreads);
                    finishLightingTraining << <numBlocks, numThreads >> > (sceneHandle, lightSamplingDist.getCopyForKernel(), lightSamplingResults.getCopyForKernel(), lightSamplingCount.getCopyForKernel(), lightSamplingNewResults.getCopyForKernel(), lightSamplingNewCount.getCopyForKernel());
                    CHECK_CUDA_ERROR("finish lighting training");

                    syncDimension(lightingQueue.tasks.data, lightingQueue.count(), maxDimension);

                    lightingQueue.setNumBlocksThreads(numBlocks, numThreads);
                    std::string genNextRayEvent = std::string("genNext ") + std::to_string(round) + " " + std::to_string(depth);
                    Timer::getInstance().timedRun(genNextRayEvent, [&]() {
                        genNextRay << <numBlocks, numThreads >> > (sceneHandle, samplerObject.getCopyForKernel(), lightingQueue.getCopyForKernel(), nextRoundRayQueue->getCopyForKernel(), depth,QTable.getCopyForKernel(), nextRayInfos.getCopyForKernel());
                    });
                }

                lightingQueue.clear();

                ++depth;
                std::swap(thisRoundRayQueue, nextRoundRayQueue);

            }

            setNumBlocksThreads(samplesCount, numBlocks, numThreads);

            addSamplesToFilm << <numBlocks, numThreads >> > (film.getCopyForKernel(), result.data, allSamples.data, samplesCount);
            CHECK_CUDA_ERROR("add sample to film");

            ++round;


        }
        //debugPrintQTable(QTable);

    }

}

/*
#include "BDPT.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"
#include "../Core/Impl.h"
#include "../Utils/Timer.h"
#include "../Utils/Utils.h"

namespace BDPT{

    enum class VertexType { Camera, Light, Surface};

    struct CameraVertex {
        float3 position;
        float3 normal;
    };
    struct LightVertex {
        const LightObject *light;
        float3 position;
        float3 normal;
    };


    using VertexVariant = Variant<CameraVertex,LightVertex,IntersectionResult>;


    class Vertex : public VertexVariant {
    public:

        VertexType type;
        Spectrum multiplier;
        bool delta = false;
        float pdfForward = 0, pdfReverse = 0;
        float3 exitantDirection;

        __host__ __device__
        Vertex() {}

        template<typename V>
        __host__ __device__
        Vertex(const V& v) :  VertexVariant(v) {}

        __host__ __device__
        Vertex(const Vertex& other) :  VertexVariant(other.value) {}


        __host__ __device__
        Vertex& operator=(const Vertex& other) {
            value = other.value;
            return *this;
        }

        __device__
        float3 getPosition() const {
            auto visitor = [&](auto& arg) -> float3 {
                return arg.position;
            };
            return visit(visitor);
        }

        __device__
        float3 getNormal() const {
            auto visitor = [&](auto& arg) -> float3 {
                return arg.normal;
            };
            return visit(visitor);
        }

        __device__
        bool isOnSurface() const { 
            return type == VertexType::Surface || (type == VertexType::Light && get<LightVertex>()->light->is<DiffuseAreaLight>()); 
        }

        __device__
        Spectrum eval(const Vertex &next, TransportMode mode) const {
            float3 incidentDirection = next.getPosition() - getPosition();
            if (lengthSquared(incidentDirection)==0) return make_float3(0,0,0);
            incidentDirection = normalize(incidentDirection);
            return get<IntersectionResult>()->bsdf.eval(incidentDirection, exitantDirection);
        }
        
        __device__
        bool isConnectible() const {
            switch (type) {
                case VertexType::Light:
                    return true; //  we don't actually support delta lights yet (Point Lights are not supported);
                case VertexType::Camera:
                    return true;
                case VertexType::Surface:
                    return !get<IntersectionResult>() -> bsdf.isDelta();
            }
            SIGNAL_ERROR("Unhandled vertex type %d in isConnectable()\n",(int)type);
            return false;
        }

        __device__
        bool isLight() const {
            return type == VertexType::Light || (type == VertexType::Surface && get<IntersectionResult>()->primitive->areaLight);
        }

        __device__
        const LightObject* getLight() const{
            if(!isLight()){
                SIGNAL_ERROR("pdfLight called on non-light vertex\n");
            }
            const LightObject *light;

            if(type==VertexType::Light){
                light = get<LightVertex>()->light;
            } 
            else{
                light = get<IntersectionResult>()->primitive->areaLight;
            }
            if(light==nullptr){
                SIGNAL_ERROR("getLight returned nullptr\n");
            }
            return light;
        }

        __device__
        bool isInfiniteLight() const {
            return type == VertexType::Light && get<LightVertex>()->light->is<EnvironmentMap>();
        }

        __device__
        Spectrum radianceEmmision(const SceneHandle &scene, const Vertex &v) const {
            if (!isLight()) return make_float3(0,0,0);
            float3 w = v.getPosition() - getPosition();
            if (lengthSquared(w) == 0) return make_float3(0,0,0);
            w = normalize(w);
            if (isInfiniteLight()) {
                // not implemented yet
                return make_float3(0,0,0);
            } else {
                const DiffuseAreaLight *light;
                IntersectionResult lightSurface;

                if(type==VertexType::Light){
                    light = get<LightVertex>()->light->get<DiffuseAreaLight>();
                    lightSurface.position = getPosition();
                    lightSurface.normal = getNormal();
                } 
                else{
                    light = get<IntersectionResult>()->primitive->areaLight->get<DiffuseAreaLight>();
                    lightSurface = *get<IntersectionResult>();
                }
                Ray rayToLight = {getPosition()+w, -w};
                return light->evaluateRay(rayToLight,lightSurface);
            }
        }

        __device__
        float convertDensity(float pdf, const Vertex &next) const {
            // Return solid angle density if _next_ is an infinite area light
            if (next.isInfiniteLight()) return pdf;
            float3 w = next.getPosition() - getPosition();
            if (lengthSquared(w) == 0) return 0;
            float invDist2 = 1.f / lengthSquared(w);
            if (next.isOnSurface())
                pdf *= abs(dot(next.getNormal(), w * sqrt(invDist2)));
            return pdf * invDist2;
        }

        __device__
        float pdf(const SceneHandle &scene, const Vertex *prev,
            const Vertex *next) const {
            if (type == VertexType::Light) return pdfLight(scene, *next);
            // Compute directions to preceding and next vertex
            float3 wn = next->getPosition() - getPosition();
            if (lengthSquared(wn) == 0) return 0;
            wn = normalize(wn);
            float3 wp;
            if (prev) {
                wp = prev->getPosition() - getPosition();
                if (lengthSquared(wp) == 0) return 0;
                wp = normalize(wp);
            } else {
                if(type!=VertexType::Camera){
                    SIGNAL_ERROR("type should be camera here\n");
                }
            }
            
            float pdf = 0, unused;
            if (type == VertexType::Camera){
                Ray ray;
                ray.origin = get<CameraVertex>()->position;
                ray.direction = wn;
                get<CameraVertex>()->camera->pdf(ray,&unused,&pdf);
            }
            else if (type == VertexType::Surface){
                pdf = get<IntersectionResult>()->bsdf.pdf(wp, wn);
            }
            return convertDensity(pdf, *next);
        }

        __device__
        float pdfLight(const SceneHandle &scene, const Vertex &v) const {
            float3 w = v.getPosition() - getPosition();
            float invDist2 = 1 / lengthSquared(w);
            w *= sqrt(invDist2);
            float pdf;
            if (isInfiniteLight()) {
                // Compute planar sampling density for infinite light sources
                //Point3f worldCenter;
                //float worldRadius;
                //scene.WorldBound().BoundingSphere(&worldCenter, &worldRadius);
                //pdf = 1 / (Pi * worldRadius * worldRadius);
                SIGNAL_ERROR("not implemented: pdf infinite light in bdpt\n");
            } 
            else {
                const LightObject *light = getLight();
                // Compute sampling density for non-infinite light sources
                float pdfPos, pdfDir;
                Ray ray;
                ray.origin = getPosition();
                ray.direction = w;
                light->sampleRayPdf(ray, getNormal(), pdfPos, pdfDir);
                pdf = pdfDir * invDist2;
            }
            if (v.isOnSurface()) pdf *= abs(dot(v.getNormal(), w));
            return pdf;
        }

        float pdfLightOrigin(const SceneHandle &scene, const Vertex &v) const {
            float3 w = v.p() - p();
            if (lengthSquared(w)) return 0.;
            w = normalize(w);
            if (isInfiniteLight()) {
                SIGNAL_ERROR("Not Implemented:pdfLightOrigin infinite\n");
            } 
            else {
                // Return solid angle density for non-infinite light sources
                float pdfPos, pdfDir, pdfChoice = 0;

                const LightObject *light = getLight();

                pdfChoice = 1.f/scene.lightsCount;

                Ray ray
                ray.origin = getPosition();
                ray.direction = w;

                light->get<DiffuseAreaLight>()->DiffuseAreaLight::sampleRayPdf(ray,gerNormal,pdfPos,pdfDir);
                return pdfPos * pdfChoice;
            }
        }

        // returns -1 if not associated to any mesh;
        __device__
        int getMeshIndex(){
            if(type==VertexType::Surface){
                return get<IntersectionResult>()->primitive->shape->meshIndex;
            }
            if(type==VertexType::Light){
                return get<LightVertex>()->light->get<DiffuseAreaLight>()->shapeIndex;
            }
        }
    };

    __device__
    bool isVisible(const SceneHandle &scene, const Vertex &v0, const Vertex &v1) {
        float distance = length(v0.getPosition()-v1.getPosition());
        Ray ray;
        ray.origin = v0.getPosition();
        ray.direction = normalize(v1.getPosition()-v0.getPosition());
        VisibilityTester tester;
        tester.ray = ray;
        tester.distanceLimit = distance;
        tester.useDistanceLimit = true;
        
        tester.sourceMeshIndex = v0.getMeshIndex();
        tester.targetMeshIndex = v0.getMeshIndex();

        return scene.testVisibility(tester);
    }

    __device__
    Spectrum G(const SceneHandle &scene, SamplerObject &sampler, const Vertex &v0, const Vertex &v1) {
        float3 d = v0.p() - v1.p();
        float g = 1 / lengthSquared(d);
        d *= sqrt(g);
        if (v0.isOnSurface()) g *= abs(dot(v0.getNormal(), d));
        if (v1.isOnSurface()) g *= abs(dot(v1.getNormal(), d));
        if(isVisible(scene,v0,v1)){
            return g;
        }
        return 0;
    }


    __device__
    float remap0(float f){
        return f != 0 ? f : 1; 
    }
    
    __device__
    float MISWeight(const Scene &scene, Vertex *lightVertices,Vertex *cameraVertices, Vertex &sampled, int s, int t,const Distribution1D &lightPdf,const std::unordered_map<const Light *, size_t> &lightToIndex) {
        if (s + t == 2) return 1;
        float sumRi = 0;
        // Temporarily update vertex properties for current strategy

        // Look up connection vertices and their predecessors
        Vertex *qs = s > 0 ? &lightVertices[s - 1] : nullptr,
        *pt = t > 0 ? &cameraVertices[t - 1] : nullptr,
        *qsMinus = s > 1 ? &lightVertices[s - 2] : nullptr,
        *ptMinus = t > 1 ? &cameraVertices[t - 2] : nullptr;

        // Update sampled vertex for $s=1$ or $t=1$ strategy
        ScopedAssignment<Vertex> a1;
        if (s == 1)
        a1 = {qs, sampled};
        else if (t == 1)
        a1 = {pt, sampled};

        // Mark connection vertices as non-degenerate
        ScopedAssignment<bool> a2, a3;
        if (pt) a2 = {&pt->delta, false};
        if (qs) a3 = {&qs->delta, false};

        // Update reverse density of vertex $\pt{}_{t-1}$
        ScopedAssignment<float> a4;
        if (pt)
        a4 = {&pt->pdfRev, s > 0 ? qs->Pdf(scene, qsMinus, *pt)
                            : pt->PdfLightOrigin(scene, *ptMinus, lightPdf,
                                                lightToIndex)};

        // Update reverse density of vertex $\pt{}_{t-2}$
        ScopedAssignment<float> a5;
        if (ptMinus)
        a5 = {&ptMinus->pdfRev, s > 0 ? pt->Pdf(scene, qs, *ptMinus)
                                : pt->PdfLight(scene, *ptMinus)};

        // Update reverse density of vertices $\pq{}_{s-1}$ and $\pq{}_{s-2}$
        ScopedAssignment<float> a6;
        if (qs) a6 = {&qs->pdfRev, pt->Pdf(scene, ptMinus, *qs)};
        ScopedAssignment<float> a7;
        if (qsMinus) a7 = {&qsMinus->pdfRev, qs->Pdf(scene, pt, *qsMinus)};

        // Consider hypothetical connection strategies along the camera subpath
        float ri = 1;
        for (int i = t - 1; i > 0; --i) {
        ri *=
        remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
        sumRi += ri;
        }

        // Consider hypothetical connection strategies along the light subpath
        ri = 1;
        for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightvertex = i > 0 ? lightVertices[i - 1].delta
                                : lightVertices[0].IsDeltaLight();
        if (!lightVertices[i].delta && !deltaLightvertex) sumRi += ri;
        }
        return 1 / (1 + sumRi);
    }

    
    __device__
    Spectrum connectBDPT( const SceneHandle &scene, Vertex *lightVertices, Vertex *cameraVertices, Vertex* previousCameraVertices, int s,int t, const CameraObject &camera, SamplerObject &sampler, float& misWeightOutput) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= samplesCount) {
            return;
        }

        Spectrum L = make_float3(0,0,0);
        // Ignore invalid connections related to infinite area lights
        if (t > 1 && s != 0 && cameraVertices[index].type == VertexType::Light)
            return make_float3(0,0,0);

        // Perform connection and write contribution to _L_
        Vertex sampled;
        if (s == 0) {
            // Interpret the camera subpath as a complete path
            const Vertex &pt = cameraVertices[index];
            if (pt.isLight()) {
                L = pt.radianceEmmision(scene, previousCameraVertices[index]) * pt.multiplier;
            }
        } else if (t == 1) {
            // Sample a point on the camera and connect it to the light subpath
            const Vertex &qs = lightVertices[index];
            if (qs.isConnectible()) {
                VisibilityTester vis;
                Vector3f wi;
                
                L = qs.multiplier * qs.eval(cameraVertices[index], TransportMode::Importance) * cameraVertices[index].multiplier;
                if (qs.isOnSurface()) L *= abs(dot(wi, qs.ns()));
                
                if (!isAllZero(L)){
                    L *= isVisible(scene,qs,cameraVertices[index]);
                } 
            }
        } else if (s == 1) {
            // Sample a point on a light and connect it to the camera subpath
            const Vertex &pt = cameraVertices[t - 1];
            if (pt.isConnectible()) {
                float lightPdf;
                VisibilityTester vis;
                vis.sourceMeshIndex = pt.getMeshIndex();

                int lightIndex = sampler.randInt(scene.lightsCount);

                const LightObject& light = scene.lights[lightIndex];
                Ray rayToLight;
                float pdf;
                float4 randomSource = sampler.rand4();

                IntersectionResult lightSurface;
                Spectrum lightWeight = light.sampleRayToPoint(intersection.position, sampler, pdf, rayToLight, vis,lightSurface);
        
                if (pdf > 0 && !isAllZero(lightWeight)) {
                    sampled = LightVertex{&light,lightSurface.position,lightSurface.normal}; 
                    sampled.multiplier = lightWeight / (pdf * lightPdf);
                    sampled.pdfFwd = sampled.PdfLightOrigin(scene, pt);
                    L = pt.multiplier * pt.eval(sampled, TransportMode::Radiance) * sampled.multiplier;
                    if (pt.isOnSurface()) L *= abs(dot(wi, pt.ns()));
                    // Only check visibility if the path would carry radiance.
                    if (!isAllZero(L)){
                        L *= scene.testVisibility(vis);
                    } 
                }
            }
        } else {
            // Handle all other bidirectional connection cases
            const Vertex &qs = lightVertices[index];
            const Vertex &pt = cameraVertices[index];
            if (qs.isConnectible() && pt.isConnectible()) {
                L = qs.multiplier * qs.eval(pt, TransportMode::Importance) * pt.eval(qs, TransportMode::Radiance) * pt.multiplier;
                if (!isAllZero(L)){
                    L *= G(scene, sampler, qs, pt);
                } 
            }
        }


        // Compute MIS weight for connection strategy
        float misWeight =
            L.IsBlack() ? 0.f : MISWeight(scene, lightVertices, cameraVertices,
                                        sampled, s, t, lightDistr, lightToIndex);
        L *= misWeight;
        if (misWeightPtr) *misWeightPtr = misWeight;
        return L;
    }
    
    BDPTIntegrator::BDPTIntegrator(int maxDepth_):maxDepth(maxDepth_) {

    }

    __global__
    void writeInitalCameraVertex(int samplesCount, Vertex* vertices, CameraSample* cameraSamples, CameraObject camera){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= samplesCount) {
            return;
        }
        vertices[index] = CameraVertex{camera.getPosition(),camera.getFront()};
    }

    std::vector<GpuArray<Vertex>> generateCameraSubPaths(const Scene& scene,const GpuArray<CameraSample>& allSamples,SamplerObject& sampler,const CameraObject& camera){
        std::vector<GpuArray<Vertex>> cameraSubPaths;

        int samplesCount = (int)allSamples.N;
        int numBlocks, numThreads;
        setNumBlocksThreads(samplesCount, numBlocks, numThreads);
        
        cameraSubPaths.emplace_back(samplesCount,false);

        writeInitalCameraVertex<<<numBlocks,numThreads>>> (samplesCount,cameraSubPaths[0].data,allSamples.data,camera);

        return cameraSubPaths;
    }

    void BDPTIntegrator::render(const Scene& scene, const CameraObject& camera, Film& film){
        int round = 0;

        while(!isFinished( scene, camera,  film)){
            GpuArray<CameraSample> allSamples = sampler->genAllCameraSamples(camera, film);

            SceneHandle sceneHandle = scene.getDeviceHandle();
    
            SamplerObject& samplerObject = *sampler;
    
            int samplesCount = (int)allSamples.N;
            int numBlocks, numThreads;
            setNumBlocksThreads(samplesCount, numBlocks, numThreads);

            sampler->prepare(samplesCount);
    
            std::vector<GpuArray<Vertex>> cameraSubPaths = generateCameraSubPaths(scene,allSamples,sampler,camera);

            ++round;
        }
    }

    
}
*/
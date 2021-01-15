#include "BDPT.h"
#include "../Samplers/SimpleSampler.h"
#include "../Utils/TaskQueue.h"
#include "../Core/Impl.h"
#include "../Utils/Timer.h"
#include "../Utils/Utils.h"

namespace BDPT{

    enum class VertexType { Camera, Light, Surface};

    struct CameraVertex {
        const CameraObject *camera;
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

    };


    /*
    struct VertexOld {
        
        
        
        float PdfLightOrigin(const Scene &scene, const Vertex &v,
                            const Distribution1D &lightDistr,
                            const std::unordered_map<const Light *, size_t>
                                &lightToDistrIndex) const {
            float3 w = v.getPosition() - getPosition();
            if (w.LengthSquared() == 0) return 0.;
            w = Normalize(w);
            if (isInfiniteLight()) {
                // Return solid angle density for infinite light sources
                return InfiniteLightDensity(scene, lightDistr, lightToDistrIndex,
                                            w);
            } else {
                // Return solid angle density for non-infinite light sources
                float pdfPos, pdfDir, pdfChoice = 0;

                // Get pointer _light_ to the light source at the vertex
                CHECK(IsLight());
                const Light *light = type == VertexType::Light
                                        ? ei.light
                                        : si.primitive->GetAreaLight();
                CHECK(light != nullptr);

                // Compute the discrete probability of sampling _light_, _pdfChoice_
                CHECK(lightToDistrIndex.find(light) != lightToDistrIndex.end());
                size_t index = lightToDistrIndex.find(light)->second;
                pdfChoice = lightDistr.DiscretePDF(index);

                light->Pdf_Le(Ray(getPosition(), w, Infinity, time()), ng(), &pdfPos, &pdfDir);
                return pdfPos * pdfChoice;
            }
        }
        
    };*/
    
    BDPTIntegrator::BDPTIntegrator(int maxDepth_):maxDepth(maxDepth_) {

    }
    void BDPTIntegrator::render(const Scene& scene, const CameraObject& camera, FilmObject& film){

    }
}
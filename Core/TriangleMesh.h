#pragma once

#include "../Core/Parameters.h"
#include "../Utils/Array.h"
#include <string>
#include <filesystem>
#include "IntersectionResult.h"
#include "../Utils/MathsCommons.h"
#include "Ray.h"
#include "../Samplers/SamplerObject.h"


class Scene;
class SceneHandle;
class Primitive;

enum class MeshShapeType{
    Irregular, Sphere, Disk
};

class TriangleMesh{
public:

    int trianglesCount;
    int verticesCount;

    bool hasVertexNormals;
    bool reverseOrientation = false; // flip vertex normals
    
    ArrayPair<float3> positions;
    ArrayPair<float3> normals;
    ArrayPair<float2> texCoords;

    ArrayPair<int3> indices;

    float surfaceArea = -1;

    MeshShapeType shapeType = MeshShapeType::Irregular;
    float4 shapeParams; //for regular shapes such as spheres, this records its defining parameters

    int meshIndex;// index of this mesh in the scene-global array of meshes
    Primitive* prim;// the enclosing primitive. This field is set during buildGpuReferences()/buildCpuReferences()

    __host__ 
    TriangleMesh();

    __host__ 
    TriangleMesh(int trianglesCount_, int verticesCount_,bool hasVertexNormals_, bool isCopyForKernel_);

    __host__
    static TriangleMesh createFromPLY(const std::string& filename,const glm::mat4& transform);

    static TriangleMesh createFromObjectDefinition(const ObjectDefinition& def, const glm::mat4& transform, const std::filesystem::path& basePath);

    static TriangleMesh createFromParams(const Parameters& params,const glm::mat4& transform, const std::filesystem::path& basePath);

    void buildCpuReferences(const SceneHandle& scene, int primIndex);

    __device__
    void buildGpuReferences(const SceneHandle& scene, int primIndex);

    __host__
    void copyToDevice();

    //TriangleMesh getCopyForKernel();

    void copyTrianglesToScene(Scene& scene,int meshID);


    __host__ __device__
    float area() const  {
        return surfaceArea;
    }

    void computeArea();

    __host__ __device__
    void getNormal(IntersectionResult& result, int triangleIndex, float u, float v) const {
#ifdef __CUDA_ARCH__
        float3* positionsData = positions.gpu.data;
        float3* normalsData = normals.gpu.data;
        int3* indicesData = indices.gpu.data;
#else
        float3* positionsData = positions.cpu.data;
        float3* normalsData = normals.cpu.data;
        int3* indicesData = indices.cpu.data;
#endif
        if (hasVertexNormals) {
            int3 thisIndices = indicesData[triangleIndex];
            float3 n0 = normalsData[thisIndices.x];
            float3 n1 = normalsData[thisIndices.y];
            float3 n2 = normalsData[thisIndices.z];
            result.normal = normalize(n0 * (1.f - u - v) + u * n1 + v * n2);
        }
        else {
            int3 thisIndices = indicesData[triangleIndex];
            float3 p0 = positionsData[thisIndices.x];
            float3 p1 = positionsData[thisIndices.y];
            float3 p2 = positionsData[thisIndices.z];
            result.normal = normalize(cross(p1 - p0, p2 - p0));
        }
        if (reverseOrientation) {
            result.normal *= -1;
        }
    }


    // if useSeenFrom is false, then the seenFrom argument is ignored.
    __device__
    IntersectionResult sample(const float3& seenFrom, SamplerObject& sampler,SamplingState& samplingState, float* outputProbability, bool useSeenFrom = true) const {
        if (shapeType == MeshShapeType::Sphere && useSeenFrom) {
            float3 center = make_float3(shapeParams.x, shapeParams.y, shapeParams.z);
            float radius = shapeParams.w;
            if (lengthSquared(seenFrom - center) > radius * radius) {
                return sampleSphere(seenFrom, sampler.rand2(samplingState), outputProbability, center, radius);
            }
        }
        
        float3* positionsData = positions.gpu.data;
        int3* indicesData = indices.gpu.data;

        IntersectionResult result;


        int triangleID =  sampler.randInt(trianglesCount,samplingState);

        float temp = sqrt(sampler.rand1(samplingState));
        float u = 1 - temp;
        float v = temp * sampler.rand1(samplingState);


        int3 thisIndices = indicesData[triangleID];
        float3 p0 = positionsData[thisIndices.x];
        float3 p1 = positionsData[thisIndices.y];
        float3 p2 = positionsData[thisIndices.z];


        result.position = p0 * u + p1 * v + p2 * (1.f - u - v);
        getNormal(result, triangleID, u, v);
            

        //if (dot(result.normal, seenFrom - result.position) < 0) {
            
        //}

        result.intersected = true;

        float3 lightToSeenFrom = seenFrom - result.position;
        float cosine = abs(dot(result.normal, normalize(lightToSeenFrom)));
        *outputProbability = lengthSquared(lightToSeenFrom) / (cosine * area());

        return result;
    }

    __device__
    IntersectionResult sampleSphere(const float3& seenFrom, float2 randomSource, float* outputProbability, const float3& center, float radius) const {

        // Sample sphere uniformly inside subtended cone

        // Compute coordinate system for sphere sampling
        float dc = length(seenFrom - center);
        float invDc = 1 / dc;
        float3 wc = (center - seenFrom) * invDc;
        float3 wcX, wcY;
        buildCoordinateSystem(wc, wcX, wcY);

        // Compute $\theta$ and $\phi$ values for sample in cone
        float sinThetaMax = radius * invDc;
        float sinThetaMax2 = sinThetaMax * sinThetaMax;
        float invSinThetaMax = 1 / sinThetaMax;
        float cosThetaMax = sqrt(max((float)0.f, 1 - sinThetaMax2));

        float2 u = randomSource;

        float cosTheta = (cosThetaMax - 1) * u.x + 1;
        float sinTheta2 = 1 - cosTheta * cosTheta;

        if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */) {
            /* Fall back to a Taylor series expansion for small angles, where
               the standard approach suffers from severe cancellation errors */
            sinTheta2 = sinThetaMax2 * u.x;
            cosTheta = sqrt(1 - sinTheta2);
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        float cosAlpha = sinTheta2 * invSinThetaMax +
            cosTheta * sqrt(max(0.f, 1.f - sinTheta2 * invSinThetaMax * invSinThetaMax));
        float sinAlpha = sqrt(max(0.f, 1.f - cosAlpha * cosAlpha));
        float phi = u.y * 2 * M_PI;

        // Compute surface normal and sampled point on sphere
        float3 nWorld = -1.f *(
            sinAlpha * std::cos(phi) * wcX + 
            sinAlpha * std::sin(phi) * wcY +
            cosTheta * wc
            );

        float3 pWorld = center + radius * nWorld;

        // Return _Interaction_ for sampled point on sphere
        IntersectionResult it;
        it.position = pWorld;
        it.normal = nWorld;
        if (reverseOrientation) it.normal *= -1;

        // Uniform cone PDF.
        *outputProbability = 1 / (2 * M_PI * (1 - cosThetaMax));

        return it;
    }


    __device__
    float pdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const {
        if (shapeType == MeshShapeType::Sphere) {
            float3 center = make_float3(shapeParams.x, shapeParams.y, shapeParams.z);
            float radius = shapeParams.w;
            if (lengthSquared(sampledRay.origin - center) > radius * radius) {
                return spherePdf(sampledRay,lightSurface, center, radius);
            }
        }
        float3 lightToSeenFrom = sampledRay.origin - lightSurface.position;
        float cosine = abs(dot(lightSurface.normal, normalize(lightToSeenFrom)));
        return lengthSquared(lightToSeenFrom) / (cosine * area());
    };

    __device__
    float spherePdf(const Ray& sampledRay, const IntersectionResult& lightSurface,const float3& center, float radius) const {
        float sinThetaMax2 = radius * radius / lengthSquared(sampledRay.origin - center);
        float cosThetaMax = sqrt(max(0.f, 1.f - sinThetaMax2));
        return 1.f / (2.f * M_PI * (1 - cosThetaMax));
    };


    void prepareForRender(Scene& scene, int meshID);
};


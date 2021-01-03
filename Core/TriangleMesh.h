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

    Primitive* prim;// index of the enclosing primitive. This field is set during buildGpuReferences()/buildCpuReferences()

    // the index of the 0th triangle of this mesh in the scene-global array of triangles.
    // this is needed when sampling the mesh.
    int globalTriangleIndexBegin;

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



    __device__
    IntersectionResult sample(const float3& seenFrom, SamplerObject& sampler, float* outputProbability) const {
#ifdef __CUDA_ARCH__
        float3* positionsData = positions.gpu.data;
        int3* indicesData = indices.gpu.data;
#else
        float3* positionsData = positions.cpu.data;
        int3* indicesData = indices.cpu.data;
#endif
        IntersectionResult result;


        int triangleID =  sampler.randInt(trianglesCount);

        float temp = sqrt(sampler.rand1());
        float u = 1 - temp;
        float v = temp * sampler.rand1();


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
    float pdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const {
        float3 lightToSeenFrom = sampledRay.origin - lightSurface.position;
        float cosine = abs(dot(lightSurface.normal, normalize(lightToSeenFrom)));
        return lengthSquared(lightToSeenFrom) / (cosine * area());
    };


    void prepareForRender(Scene& scene, int meshID);
};


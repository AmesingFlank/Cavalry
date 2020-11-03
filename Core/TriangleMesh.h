#pragma once

#include "../Core/Parameters.h"
#include "../Utils/Array.h"
#include <string>
#include <filesystem>
#include "IntersectionResult.h"
#include "../Utils/MathsCommons.h"
#include "Ray.h"


class Scene;

class TriangleMesh{
public:

    int trianglesCount;
    int verticesCount;

    bool hasVertexNormals;
    
    ArrayPair<float3> positions;
    ArrayPair<float3> normals;
    ArrayPair<float2> texCoords;

    ArrayPair<int3> indices;

    float surfaceArea;

    __host__ 
    TriangleMesh();

    __host__ 
    TriangleMesh(int trianglesCount_, int verticesCount_,bool hasVertexNormals_, bool isCopyForKernel_);

    __host__ __device__
    const TriangleMesh* getID() const{
        return this;
    }

    __host__
    static TriangleMesh createFromPLY(const std::string& filename,const glm::mat4& transform);

    static TriangleMesh createFromObjectDefinition(const ObjectDefinition& def, const glm::mat4& transform, const std::filesystem::path& basePath);

    static TriangleMesh createFromParams(const Parameters& params,const glm::mat4& transform, const std::filesystem::path& basePath);

    __host__
    void copyToDevice();

    TriangleMesh getCopyForKernel();

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
            result.normal = normalize(cross(p2 - p0, p1 - p0));
        }
    }

    __host__ __device__
    IntersectionResult sample(const float4& randomSource,float* outputProbability) const {
        
        int triangleID = round(randomSource.z*(trianglesCount-1));
        float temp = sqrt(randomSource.x);
        float u = 1-temp;
        float v = temp*randomSource.y;

#ifdef __CUDA_ARCH__
        float3* positionsData = positions.gpu.data;
        int3* indicesData = indices.gpu.data;
#else
        float3* positionsData = positions.cpu.data;
        int3* indicesData = indices.cpu.data;
#endif
        int3 thisIndices = indicesData[triangleID];
        float3 p0 = positionsData[thisIndices.x];
        float3 p1 = positionsData[thisIndices.y];
        float3 p2 = positionsData[thisIndices.z];

        

        IntersectionResult result;
        result. position = p0*u+p1*v+p2*(1.f-u-v);
        getNormal(result,triangleID,u,v);

        
        result.intersected = true;

        *outputProbability = 1.f/surfaceArea;

        return result;
    }

    __host__ __device__
    IntersectionResult sample(const float3& seenFrom, const float4& randomSource, float* outputProbability) const {
        float dummy;
        IntersectionResult result = sample(randomSource, &dummy);
        float3 lightToRay = seenFrom - result.position;
        float cosine = abs(dot(result.normal, normalize(lightToRay)));
        *outputProbability = lengthQuared(lightToRay) / (cosine * area());
        return result;
    }

    void prepareForRender(Scene& scene,int meshID)  {
        computeArea();
        copyToDevice();
        copyTrianglesToScene(scene,meshID);
    };

};


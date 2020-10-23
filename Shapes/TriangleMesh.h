#pragma once

#include "../Core/Shape.h"
#include "../Core/Parameters.h"
#include "../Utils/Array.h"
#include <string>
#include <filesystem>

__host__ __device__ 
inline bool rayTriangleIntersection(IntersectionResult& result, const Ray& r,
	const float3 &v0,
	const float3 &edge1,
	const float3 &edge2,
    float &u,
    float &v)
{

	float3 tvec = r.origin - v0;
	float3 pvec = cross(r.direction, edge2);

	float det = dot(edge1, pvec);

	det = 1.f/det;

	u = dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f){
        result.intersected = false;
        return false;
    }

	float3 qvec = cross(tvec, edge1);

	v = dot(r.direction, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f){
        result.intersected = false;
        return false;
    }

    

    result.distance = dot(edge2, qvec) * det;

    if (result.distance <= 0.f) {
        result.intersected = false;
        return false;
    }

    result.position = r.positionAtDistance(result.distance);
    result.intersected = true;

    return true;
}


class TriangleMesh: public Shape{
public:

    int trianglesCount;
    int verticesCount;

    bool hasVertexNormals;
    
    ArrayPair<float3> positions;
    ArrayPair<float3> normals;
    ArrayPair<float2> texCoords;

    ArrayPair<int3> indices;

    __host__ 
    TriangleMesh();

    __host__ 
    TriangleMesh(int trianglesCount_, int verticesCount_,bool hasVertexNormals_, bool isCopyForKernel_);

    __host__
    static TriangleMesh createFromPLY(const std::string& filename,const glm::mat4& transform);

    static TriangleMesh createFromParams(const Parameters& filename,const glm::mat4& transform, const std::filesystem::path& basePath);

    __host__
    void copyToDevice();

    TriangleMesh getCopyForKernel();


    __host__ __device__
    void getNormal(IntersectionResult& result, int triangleIndex,float u, float v) const{
#ifdef __CUDA_ARCH__
        float3* positionsData = positions.gpu.data;
        float3* normalsData = normals.gpu.data;
        int3* indicesData = indices.gpu.data;
#else
        float3* positionsData = positions.cpu.data;
        float3* normalsData = normals.cpu.data;
        int3* indicesData = indices.cpu.data;
#endif
        if(hasVertexNormals){
            int3 thisIndices = indicesData[triangleIndex];
            float3 n0 = normalsData[thisIndices.x];
            float3 n1 = normalsData[thisIndices.y];
            float3 n2 = normalsData[thisIndices.z];
            result.normal= normalize(n0*(1.f-u-v) + u*n1+v*n2);
        }
        else{
            int3 thisIndices = indicesData[triangleIndex];
            float3 p0 = positionsData[thisIndices.x];
            float3 p1 = positionsData[thisIndices.y];
            float3 p2 = positionsData[thisIndices.z];
            result.normal = normalize(cross(p2-p0,p1-p0));
        }
    }



    __host__ __device__
    virtual bool intersect(IntersectionResult& result, const Ray& ray) const override{
        // naive bruteforce
#ifdef __CUDA_ARCH__
        float3* positionsData = positions.gpu.data;
        float3* normalsData = normals.gpu.data;
        int3* indicesData = indices.gpu.data;
#else
        float3* positionsData = positions.cpu.data;
        float3* normalsData = normals.cpu.data;
        int3* indicesData = indices.cpu.data;
#endif

        result.intersected = false;

        for(int i = 0;i<trianglesCount;++i){
            int3 thisIndices = indicesData[i];
            float3 v0 = positionsData[thisIndices.x];
            float3 edge1 = positionsData[thisIndices.y]-v0;
            float3 edge2 = positionsData[thisIndices.z]-v0;
            IntersectionResult thisResult;
            float u,v;
            if(rayTriangleIntersection(thisResult,ray,v0,edge1,edge2,u,v)){
                if(!result.intersected || result.distance > thisResult.distance){
                    getNormal(thisResult,i,u,v);
                    result = thisResult;
                }
            }
        }

        if (dot(result.normal, ray.direction) > 0) {
            result.normal *= -1;
        }
        
        return result.intersected;


    }

};


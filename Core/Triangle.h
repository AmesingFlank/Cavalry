#pragma once

#include "TriangleMesh.h"


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

    if (det == 0) {
        result.intersected = false;
        return false;
    }

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


class Triangle {
public:

    int meshIndex;
    int triangleIndex;

    const TriangleMesh* mesh;
    const Primitive* prim;

    Triangle(int meshIndex_,int triangleIndex_):meshIndex(meshIndex_),triangleIndex(triangleIndex_){

    }

     __host__ __device__
    bool intersect(IntersectionResult& result, const Ray& ray) const{

#ifdef __CUDA_ARCH__
        float3* positionsData = mesh->positions.gpu.data;
        float3* normalsData = mesh->normals.gpu.data;
        int3* indicesData = mesh->indices.gpu.data;
#else
        float3* positionsData = mesh->positions.cpu.data;
        float3* normalsData = mesh->normals.cpu.data;
        int3* indicesData = mesh->indices.cpu.data;
#endif

        int3 thisIndices = indicesData[triangleIndex];
        float3 v0 = positionsData[thisIndices.x];
        float3 edge1 = positionsData[thisIndices.y]-v0;
        float3 edge2 = positionsData[thisIndices.z]-v0;
        float u,v;

        if(rayTriangleIntersection(result,ray,v0,edge1,edge2,u,v)){
            if(mesh->hasVertexNormals){
                float3 n0 = normalsData[thisIndices.x];
                float3 n1 = normalsData[thisIndices.y];
                float3 n2 = normalsData[thisIndices.z];
                result.normal= normalize(n0*(1.f-u-v) + u*n1+v*n2);
            }
            else{
                result.normal = normalize(cross(edge1,edge2));
            }
            if (dot(result.normal, ray.direction) > 0) {
                result.normal *= -1;
            }
            result.primitive = prim;
            return true;
        }
        return false;

        

    }

   


    void buildCpuReferences(const SceneHandle& scene);

    __device__
    void buildGpuReferences(const SceneHandle& scene);


};


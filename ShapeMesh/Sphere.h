#pragma once

#include "../Core/TriangleMesh.h"
#include <vector>
#include "../Utils/MathsCommons.h"


inline TriangleMesh createSphereMesh(float radius, const glm::mat4& transform) {

    std::vector<int3> indicesA,indicesB;
    std::vector<int3>* currIndices = &indicesA;
    std::vector<int3>* lastIndices = &indicesB;

    std::vector<float3> positions;



    // initially an octahedron
    positions.push_back(make_float3(1, 0, 0));
    positions.push_back(make_float3(-1, 0, 0));
    positions.push_back(make_float3(0, 1, 0));
    positions.push_back(make_float3(0, -1, 0));
    positions.push_back(make_float3(0, 0, 1));
    positions.push_back(make_float3(0, 0, -1));

    currIndices->push_back(make_int3(0, 2, 4));
    currIndices->push_back(make_int3(0, 2, 5));
    currIndices->push_back(make_int3(0, 3, 4));
    currIndices->push_back(make_int3(0, 3, 5));
    currIndices->push_back(make_int3(1, 2, 4));
    currIndices->push_back(make_int3(1, 2, 5));
    currIndices->push_back(make_int3(1, 3, 4));
    currIndices->push_back(make_int3(1, 3, 5));


    for (int i = 0; i < 5; ++i) {

        std::swap(currIndices, lastIndices);

        // run subdivision
        currIndices->clear();

        for (int3 triangle : *lastIndices) {
            int i0 = triangle.x;
            int i1 = triangle.y;
            int i2 = triangle.z;

            float3 p0 = positions[i0];
            float3 p1 = positions[i1];
            float3 p2 = positions[i2];

            float3 p01 = normalize((p0 + p1) * 0.5f);
            positions.push_back(p01);
            int i01 = positions.size() - 1;

            float3 p02 = normalize((p0 + p2) * 0.5f);
            positions.push_back(p02);
            int i02 = positions.size() - 1;

            float3 p12 = normalize((p1 + p2) * 0.5f);
            positions.push_back(p12);
            int i12 = positions.size() - 1;

            currIndices->push_back(make_int3(i01,i02,i12));

            currIndices->push_back(make_int3(i0,i01,i02));
            currIndices->push_back(make_int3(i1, i01, i12));
            currIndices->push_back(make_int3(i2, i02, i12));
        }
    }


    std::vector<float3> normals;
    glm::mat3 normalMat = getTransformForNormal(transform);
    for (float3& pos : positions) {
        normals.push_back(normalize(normalMat * pos));
        pos = apply(transform, pos * radius);
    }


    int trianglesCount = currIndices->size();
    int verticesCount = positions.size();


    TriangleMesh mesh(trianglesCount, verticesCount, true, false);
    mesh.positions.cpu = positions;
    mesh.normals.cpu = normals;
    mesh.indices.cpu = *currIndices;
    mesh.definitelyWaterTight = true;


    return mesh;

}

inline TriangleMesh createSphereMeshUV(float radius, const glm::mat4& transform){
    std::cout << "creating sphere mesh" << std::endl;
    int thetaSegments = 360;
    int phiSegments = 180;

    int trianglesCount = thetaSegments * phiSegments * 2;
    int verticesCount = thetaSegments * (phiSegments+1);

    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<int3> indices;

    glm::mat3 normalMat = getTransformForNormal(transform);

    for(int thetaID = 0;thetaID < thetaSegments;++thetaID){
        for(int phiID = 0 ;phiID <= phiSegments;++phiID){
            float theta = 2.f * M_PI * (float)thetaID / (float)thetaSegments;
            float phi = M_PI * (float)phiID / (float)phiSegments;
            float3 pos = make_float3(
                sin(theta) * sin(phi),
                cos(phi),
                cos(theta) * sin(phi)
            );
            float3 normal = pos;

            positions.push_back(apply(transform,pos*radius));
            normals.push_back(normalMat * normal);

            if(phiID < phiSegments){
                int thisIndex = positions.size()-1;
                int downIndex = thisIndex + 1;
                int rightIndex = (thisIndex + phiSegments + 1) % verticesCount;
                int downRightIndex = rightIndex + 1;
                

                int3 triangle0 = make_int3(thisIndex,downIndex,downRightIndex);
                int3 triangle1 = make_int3(thisIndex,downRightIndex,rightIndex);

                if (phiID == 0) {
                    indices.push_back(triangle0);
                    indices.push_back(triangle0);
                }
                else if (phiID == phiSegments - 1) {
                    indices.push_back(triangle1);
                    indices.push_back(triangle1);
                }
                else {
                    indices.push_back(triangle0);
                    indices.push_back(triangle1);
                }
            }
        }
    }

    if(positions.size()!=verticesCount){
        SIGNAL_ERROR("positions size wrong");
    }
    if(normals.size()!=verticesCount){
        SIGNAL_ERROR("normals size wrong");
    }
    if(indices.size()!=trianglesCount){
        SIGNAL_ERROR("indices size wrong %d %d",indices.size(),trianglesCount);
    }
    TriangleMesh mesh(trianglesCount,verticesCount,true,false);
    mesh.positions.cpu = positions;
    mesh.normals.cpu = normals;
    mesh.indices.cpu = indices;
    mesh.definitelyWaterTight = true;

    mesh.computeArea(); std::cout << "sphere area " << mesh.area() <<"   "<<radius<< std::endl;
    std::cout << "created sphere mesh " << std::endl;
    return mesh;
}